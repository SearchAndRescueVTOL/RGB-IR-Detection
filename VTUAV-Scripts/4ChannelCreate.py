import os
from PIL import Image
import numpy as np

# Define the source and destination directories
rgb_train_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/train/images'
ir_train_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/train/ir'
rgb_val_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/validation/images'
ir_val_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/validation/ir'

train_output_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/train/4C'
val_output_dir = '/home/ubuntu/persistent/vtuav-det/VTUAV-det/data/validation/4C'

# Create the output directories if they do not exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Function to add IR as a 4th channel to the RGB image and save as .tiff
def create_4c_image(rgb_path, ir_path, output_path):
    # Open the RGB image
    rgb_image = Image.open(rgb_path).convert('RGB')
    # Open the IR image (assuming grayscale, convert to 2D array)
    ir_image = Image.open(ir_path).convert('L')
    
    # Convert both images to numpy arrays
    rgb_array = np.array(rgb_image)
    ir_array = np.array(ir_image)
    
    # Ensure the IR array has the same height and width as the RGB image
    if rgb_array.shape[0] != ir_array.shape[0] or rgb_array.shape[1] != ir_array.shape[1]:
        raise ValueError(f"Image size mismatch: RGB {rgb_array.shape} vs IR {ir_array.shape}")
    
    # Stack the IR array as a 4th channel
    ir_array_4c = np.expand_dims(ir_array, axis=-1)  # Convert 2D to 3D (height, width, 1)
    new_image_array = np.concatenate((rgb_array, ir_array_4c), axis=-1)  # Add IR as 4th channel
    
    # Convert the new 4-channel image to a PIL Image
    new_image = Image.fromarray(new_image_array)
    
    # Save the new image as .tiff
    new_image.save(output_path, format='TIFF')

# Function to process images in a given directory
def process_images(rgb_dir, ir_dir, output_dir):
    # Get all the image filenames (assuming the images have the same names)
    for rgb_filename in os.listdir(rgb_dir):
        rgb_path = os.path.join(rgb_dir, rgb_filename)
        ir_path = os.path.join(ir_dir, rgb_filename)  # IR images should have the same name as RGB images
        
        if os.path.exists(ir_path):  # Ensure corresponding IR image exists
            output_path = os.path.join(output_dir, rgb_filename.replace('.jpg', '.tiff'))  # Save as .tiff
            try:
                # Create 4C image and save it
                create_4c_image(rgb_path, ir_path, output_path)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {rgb_filename}: {e}")

# Process the training images
process_images(rgb_train_dir, ir_train_dir, train_output_dir)

# Process the validation images
process_images(rgb_val_dir, ir_val_dir, val_output_dir)
