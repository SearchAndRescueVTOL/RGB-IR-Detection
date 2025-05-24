import json
import os
from det_solver import DummyDetectionDataset
def generate_dummy_coco_json(dataset, save_path):
    images = []
    annotations = []
    categories = [{"id": i, "name": f"class_{i}"} for i in range(dataset.num_classes)]
    ann_id = 1

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        image_info = {
            "id": idx,
            "width": dataset.image_size,
            "height": dataset.image_size,
            "file_name": f"{idx}.jpg"  # dummy filename
        }
        images.append(image_info)

        for i in range(len(target["boxes"])):
            box = target["boxes"][i]
            x0, y0, x1, y1 = box.tolist()
            w = x1 - x0
            h = y1 - y0
            coco_box = [x0 * dataset.image_size, y0 * dataset.image_size,
                        w * dataset.image_size, h * dataset.image_size]

            ann = {
                "id": ann_id,
                "image_id": idx,
                "category_id": target["labels"][i].item(),
                "bbox": coco_box,
                "area": w * h * (dataset.image_size ** 2),
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(save_path, "w") as f:
        json.dump(coco_dict, f)
dataset = DummyDetectionDataset(num_samples=200, num_classes=4)
generate_dummy_coco_json(dataset, "dummy_coco_annotations.json")
