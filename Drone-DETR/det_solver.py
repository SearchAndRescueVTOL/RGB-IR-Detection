import time 
import json
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch 
from dist import *
from model import DETR_Backbone, DETR_Neck, DroneDETR
from solver import BaseSolver
from det_engine import train_one_epoch, GradScaler, Warmup, ExponentialMovingAverage, evaluate
from types import SimpleNamespace
from criterion import SetCriterion
from matcher import HungarianMatcher
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from cocoEval import CocoEvaluator
from faster_coco_eval import COCO
import numpy as np
import random
import os
import sys
import tempfile
import torch.distributed as dist
from postProcess import RTDETRPostProcessor
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class DummyDetectionDataset(Dataset):
    def __init__(self, num_samples=100, image_size=640, num_channels=4, num_classes=4, max_boxes=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.max_boxes = max_boxes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(self.num_channels, self.image_size, self.image_size)

        num_boxes = random.randint(1, self.max_boxes)
        boxes = []
        labels = []

        for _ in range(num_boxes):
            cx, cy = np.random.rand(2)
            w, h = np.random.rand(2) * 0.5

            x0 = np.clip(cx - w / 2, 0, 1)
            y0 = np.clip(cy - h / 2, 0, 1)
            x1 = np.clip(cx + w / 2, 0, 1)
            y1 = np.clip(cy + h / 2, 0, 1)

            if x1 - x0 <= 1e-6 or y1 - y0 <= 1e-6:
                continue  # filter out tiny or inverted boxes

            box = [x0, y0, x1, y1]
            if np.isnan(box).any() or np.isinf(box).any():
                continue

            boxes.append(box)
            labels.append(random.randint(0, self.num_classes - 1))

        if not boxes:
            boxes = [[0.1, 0.1, 0.2, 0.2]]
            labels = [0]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Final assertion to catch errors early
        assert (boxes[:, 2:] >= boxes[:, :2]).all(), f"Bad boxes: {boxes}"

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([640,640])
        }

        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)
    return images, list(targets)

class DetSolver(BaseSolver):
    def __init__(self, model, cfg, criterion, train_dataloader, val_dataloader, device, lr, weight_decay, epochs, log_dir, postprocessor, evaluator):
        super().__init__(cfg)
        self.model = model
        self.postprocessor=postprocessor
        self.evaluator = evaluator
        self.cfg = cfg
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(params = self.model.parameters(), lr=lr, weight_decay = weight_decay)
        self.num_epochs = epochs
        self.ema = ExponentialMovingAverage(model, 0.9999)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=400, eta_min = 0)
        self.lr_warmup_scheduler= Warmup(self.lr_scheduler, 20)
        self.writer = SummaryWriter(log_dir)
        self.last_epoch = 0
        self.output_dir = "./outputs"
        # Initialize other necessary attributes
    def fit(self):
        print("Start training")
        self.model.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, self.num_epochs):

            # self.train_dataloader.sampler.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=5, 
                print_freq=1, 
                ema=self.ema, 
                scaler=GradScaler(), 
                lr_warmup_scheduler=self.lr_warmup_scheduler, # warmup for 5-10% of total epochs
                writer=self.writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [f"{self.output_dir}/last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(f"{self.output_dir}/checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    save_on_master(self.ema.module.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            # TODO 
            for k in test_stats:
                if self.writer and is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)
            
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    save_on_master(self.state_dict(), f"{self.output_dir}/best.pth")

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and is_main_process():
                with open(f"{self.output_dir}/log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    os.makedirs(f"{self.output_dir}/eval", exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    f"{self.output_dir}/eval/{name}")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    def val(self, ):
        self.eval()
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)
                
        if self.output_dir:
            save_on_master(coco_evaluator.coco_eval["bbox"].eval, f"{self.output_dir}/eval.pth")
        
        return
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    backbone = DETR_Backbone(4).to(device)
    neck = DETR_Neck().to(device)
    model = DroneDETR(backbone, neck).to(device)
    
    losses = ['focal', 'boxes', 'cardinality']
    weight_dict = {
        'loss_focal': 1.0,
        'loss_bbox': 5.0,
        'cardinality_error': 2.0,
    }
    weight_dict_matcher = {
        "cost_class" : 1.0,
        "cost_bbox": 1.0,
        "cost_giou": 1.0
    }
    matcher = HungarianMatcher(weight_dict=weight_dict_matcher, use_focal_loss=True).to(device)
    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, losses=losses, num_classes=4).to(device)
    train_dataset = DummyDetectionDataset(num_samples=200, num_classes=4)
    val_dataset = DummyDetectionDataset(num_samples=50, num_classes=4)
    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size = 8, collate_fn=collate_fn)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=400, eta_min = 0)
    postProc = RTDETRPostProcessor(4, True)
    coco_gt = COCO("dummy_coco_annotations.json")
    evaluator=CocoEvaluator(coco_gt= coco_gt, iou_types=["bbox"])
    cfg = SimpleNamespace(checkpoint_freq=10,
                            train_dataloader=train_loader,
                            val_dataloader=val_loader,
                            device=device,
                            criterion=criterion,
                            optimizer=optim,
                            num_epochs=100,
                            ema=ExponentialMovingAverage(model, 0.9999).to(device),
                            lr_scheduler=lr_sched,  # Will be set after optimizer
                            lr_warmup_scheduler=Warmup(lr_sched, 20),  # Will be set after scheduler
                            writer=SummaryWriter("./logs"),
                            postprocessor=postProc
                          )
    solver = DetSolver(model, cfg, criterion, train_loader, val_loader, device, lr=1e-4, weight_decay = 1e-4, epochs = 100, log_dir="./logs", postprocessor=postProc, evaluator=evaluator)
    solver.fit()
    # cleanup()