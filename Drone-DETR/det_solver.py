import time 
import json
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch 
from dist import *

from _solver import BaseSolver
from det_engine import train_one_epoch, GradScaler, Warmup, ExponentialMovingAverage
class DetSolver(BaseSolver):
    def __init__(self, model, cfg, criterion, train_dataloader, val_dataloader, device, lr, weight_decay, epochs, log_dir):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(lr=lr, weight_decay = weight_decay)
        self.num_epochs = epochs
        self.ema = ExponentialMovingAverage(model, 0.9999)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=400, eta_min = 0)
        self.lr_warmup_scheduler= Warmup(self.lr_scheduler, 20)
        self.writer = SummaryWriter(log_dir)
        # Initialize other necessary attributes
    def fit(self, log_dir):
        print("Start training")
        self.model.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, self.num_epochs):

            self.train_dataloader.set_epoch(epoch)
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
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    save_on_master(self.ema.module.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            # test_stats, coco_evaluator = evaluate(
            #     module, 
            #     self.criterion, 
            #     self.postprocessor, 
            #     self.val_dataloader, 
            #     self.evaluator, 
            #     self.device
            # )

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
                    save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))