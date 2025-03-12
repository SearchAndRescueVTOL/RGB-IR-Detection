import os


dataset_root = "blended_dataset"
images_dir = os.path.join(dataset_root, "images")
labels_dir = os.path.join(dataset_root, "labels")
output_labels_dir = os.path.join(dataset_root, "labels_yolo")


os.makedirs(output_labels_dir, exist_ok=True)

img_width, img_height = 640, 640


splits = ["train", "test"]

for split in splits:
    label_split_dir = os.path.join(labels_dir, split)
    output_label_split_dir = os.path.join(output_labels_dir, split)
    os.makedirs(output_label_split_dir, exist_ok=True)

   
    for label_file in os.listdir(label_split_dir):
        label_path = os.path.join(label_split_dir, label_file)
        output_label_path = os.path.join(output_label_split_dir, label_file)

        with open(label_path, "r") as f_in, open(output_label_path, "w") as f_out:
            for line in f_in:
                fields = line.strip().split("\t")

                if len(fields) < 5 or fields[4] == "": 
                    continue  

                x1, y1, x2, y2 = map(int, fields[:4])

                
                cx = ((x1 + x2) / 2) / img_width 
                cy = ((y1 + y2) / 2) / img_height 
                w = (x2 - x1) / img_width  
                h = (y2 - y1) / img_height 

                f_out.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

print("YOLO label conversion complete.")

data_yaml_path = os.path.join(dataset_root, "data.yaml")
with open(data_yaml_path, "w") as yaml_file:
    yaml_content = f"""train: {images_dir}/train
val: {images_dir}/test
nc: 1
names: ["person"]
"""
    yaml_file.write(yaml_content)

print("data.yaml created successfully.")
