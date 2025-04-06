import json
import os

def coco_to_yolo(coco_data, output_dir, images_dir):
    """
    Convert COCO-style annotations to YOLO format.

    Args:
    - coco_data: The loaded COCO JSON data.
    - output_dir: Directory to save the YOLO label files.
    - images_dir: Directory containing the image files.
    """
    categories = {category['id']: category['name'] for category in coco_data['categories']}
    images = {image['id']: image for image in coco_data['images']}
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        image = images[image_id]
        width, height = image['width'], image['height']
        
        # Convert bounding box to YOLO format
        x_min, y_min, box_width, box_height = annotation['bbox']
        
        # Calculate the center and normalize
        center_x = (x_min + box_width / 2) / width
        center_y = (y_min + box_height / 2) / height
        norm_width = box_width / width
        norm_height = box_height / height
        
        # Get the category ID and map to class label
        category_id = annotation['category_id']
        class_id = category_id - 1  # YOLO format assumes 0-indexed class IDs

        # Prepare the YOLO format line
        yolo_line = f"{class_id} {center_x} {center_y} {norm_width} {norm_height}\n"

        # Write to the appropriate label file for the image
        label_file = os.path.join(output_dir, f"{image['file_name'].split('.')[0]}.txt")
        with open(label_file, 'a') as file:
            file.write(yolo_line)

def process_json_file(json_file, output_dir, images_dir):
    """
    Load the COCO JSON file and process annotations to YOLO format.

    Args:
    - json_file: Path to the COCO JSON file.
    - output_dir: Directory to save the YOLO label files.
    - images_dir: Directory containing the image files.
    """
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    coco_to_yolo(coco_data, output_dir, images_dir)

if __name__ == "__main__":
    # Paths to the COCO JSON files and where to store YOLO labels
    train_json = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/train_ir.json'
    val_json = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/val_ir.json'

    # Directories to save labels
    train_labels_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/train/labels'
    val_labels_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/validation/labels'
    
    # Directories where images are stored
    train_images_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/train/images'
    val_images_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/validation/images'

    # Convert training data
    process_json_file(train_json, train_labels_dir, train_images_dir)

    # Convert validation data
    process_json_file(val_json, val_labels_dir, val_images_dir)

    print("✅ Conversion complete!")
