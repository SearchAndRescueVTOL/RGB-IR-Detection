import os
import cv2
import numpy as np

dataset_root = "NII_CU_MAPD_dataset/4-channel/images"
output_dir = "blended_dataset/images"

os.makedirs(output_dir, exist_ok=True)

splits = ["train", "test"]

for split in splits:
    rgb_dir = os.path.join(dataset_root, "rgb", split)
    ir_dir = os.path.join(dataset_root, "thermal", split)
    output_split_dir = os.path.join(output_dir, split)

    os.makedirs(output_split_dir, exist_ok=True)

    image_filenames = sorted(os.listdir(rgb_dir)) 
    
    for img_name in image_filenames:
        rgb_path = os.path.join(rgb_dir, img_name)
        ir_path = os.path.join(ir_dir, img_name)

        if not os.path.exists(ir_path):
            print(f"Skipping {img_name} (IR image missing)")
            continue

        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        target_size = (640, 640)
        rgb_img = cv2.resize(rgb_img, target_size)
        ir_img = cv2.resize(ir_img, target_size)

    
        ir_img = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX)

        ir_img = np.expand_dims(ir_img, axis=2)

        blended_img = np.concatenate((rgb_img, ir_img), axis=2)

        # Save output image
        output_path = os.path.join(output_split_dir, img_name)
        cv2.imwrite(output_path, blended_img)

print("Blended dataset creation complete.")
