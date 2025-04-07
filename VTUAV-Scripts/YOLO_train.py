import sys
from ultralytics4channel import YOLO
if __name__ == "__main__":
    model = YOLO("./yolo8s.yaml")
    results = model.train(data="data.yaml", epochs=100, imgsz=640, device=[0, 1])