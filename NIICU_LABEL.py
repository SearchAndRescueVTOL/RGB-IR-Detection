import os

dataset_root = "NII_CU_MAPD_dataset/4-channel/images"
output_dir = "blended_dataset/labels"

os.makedirs(output_dir, exist_ok=True)

splits = ["train", "val"]

original_width, original_height = 2706, 1980 
new_width, new_height = 640, 640


scale_x = new_width / original_width
scale_y = new_height / original_height

for split in splits:
    label_dir = os.path.join(dataset_root, "labels", split)
    if split == "val":
        output_split_dir = os.path.join(output_dir, "test")
    else:
        output_split_dir = os.path.join(output_dir, split)

    os.makedirs(output_split_dir, exist_ok=True)

    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        output_label_path = os.path.join(output_split_dir, label_file)

        with open(label_path, "r") as f_in, open(output_label_path, "w") as f_out:
            for line in f_in:
                fields = line.strip().split("\t")
                x1, y1, x2, y2 = map(int, fields[:4])

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                new_line = f"{x1}\t{y1}\t{x2}\t{y2}\t" + "\t".join(fields[4:]) + "\n"
                f_out.write(new_line)

print("Label resizing complete.")
