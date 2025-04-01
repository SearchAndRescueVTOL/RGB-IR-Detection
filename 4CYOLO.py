import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO  
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model given configuration and weight files."""
        model = YOLO('yolov8s.yaml')
        return model
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        pass
trainer = CustomTrainer(overrides={'model': 'yolov8s.yaml', 'data': 'data.yaml'})
trainer.train()

# model = YOLO('yolov8s.pt')

# first_conv = model.model.model[0].conv

# new_conv = nn.Conv2d(
#     in_channels=4,  # Change input channels to 4
#     out_channels=first_conv.out_channels, 
#     kernel_size=first_conv.kernel_size,
#     stride=first_conv.stride,
#     padding=first_conv.padding,
#     bias=first_conv.bias is not None
# )


# new_conv.weight[:, :3, :, :].data = first_conv.weight.data.clone()  
# new_conv.weight[:, 3:, :, :].data.zero_() 
# model.model.model[0].conv = new_conv


# # Configure the d-taset in YOLO format;
# model.train(data="data.yaml", epochs=50, imgsz=640, batch=64)

