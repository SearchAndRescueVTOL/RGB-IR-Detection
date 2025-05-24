import os
import shutil

def move_images(label_dir, source_rgb_dir, source_ir_dir, target_rgb_dir, target_ir_dir):
    """
    Move the corresponding RGB and IR images based on the filenames in label files.
    
    Args:
    - label_dir: Directory containing label files (train/labels or validation/labels).
    - source_rgb_dir: Directory containing source RGB images.
    - source_ir_dir: Directory containing source IR images.
    - target_rgb_dir: Target directory for RGB images (train/images/rgb or validation/images/rgb).
    - target_ir_dir: Target directory for IR images (train/ir or validation/ir).
    """
    # Ensure target directories exist
    os.makedirs(target_rgb_dir, exist_ok=True)
    os.makedirs(target_ir_dir, exist_ok=True)
    
    # Loop through all label files in the label directory
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            # Extract the image base name from the label file name (assumes same base name)
            image_name = label_file.replace('.txt', '')
            
            # Define the source image paths (RGB and IR)
            rgb_image_path = os.path.join(source_rgb_dir, image_name + '.jpg')  # Assuming .jpg extension
            ir_image_path = os.path.join(source_ir_dir, image_name + '.jpg')   # Assuming .jpg extension
            
            # Define the target image paths
            target_rgb_image_path = os.path.join(target_rgb_dir, image_name + '.jpg')
            target_ir_image_path = os.path.join(target_ir_dir, image_name + '.jpg')
            
            # Check if RGB image exists and move it
            if os.path.exists(rgb_image_path):
                shutil.copy(rgb_image_path, target_rgb_image_path)
                print(f"Moved RGB image: {rgb_image_path} to {target_rgb_image_path}")
            else:
                print(f"RGB image not found: {rgb_image_path}")
            
            # Check if IR image exists and move it
            if os.path.exists(ir_image_path):
                shutil.copy(ir_image_path, target_ir_image_path)
                print(f"Moved IR image: {ir_image_path} to {target_ir_image_path}")
            else:
                print(f"IR image not found: {ir_image_path}")

def main():
    # Paths to the directories
    train_label_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/train/labels'
    val_label_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/validation/labels'

    source_train_rgb_dir = '/home/ubuntu/persistent/vtuav_images/train/rgb'
    source_train_ir_dir = '/home/ubuntu/persistent/vtuav_images/train/ir'
    source_val_rgb_dir = '/home/ubuntu/persistent/vtuav_images/test/rgb'
    source_val_ir_dir = '/home/ubuntu/persistent/vtuav_images/test/ir'

    target_train_rgb_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/train/images'
    target_train_ir_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/train/ir'
    target_val_rgb_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/validation/images'
    target_val_ir_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/validation/ir'

    # Move train images
    move_images(train_label_dir, source_train_rgb_dir, source_train_ir_dir, target_train_rgb_dir, target_train_ir_dir)

    # Move validation images
    move_images(val_label_dir, source_val_rgb_dir, source_val_ir_dir, target_val_rgb_dir, target_val_ir_dir)

if __name__ == "__main__":
    main()
    print("✅ Image movement complete!")
