import os
import random
import shutil
from PIL import Image, ImageFilter
import albumentations as A
import cv2
import numpy as np
from math import floor

'''
    This script automates the process of moving image and YOLO label data from the combined_data folder
    into existing training, validation, and testing folders. It can blur and rotate images, while ensuring
    that the images and their corresponding YOLO labels are moved together. The data is split by a ratio
    of 80% for training, 10% for validation, and 10% for testing, and the new data is added to the existing
    data in the train, val, and test folders.

    How to use:
        Move files you want to move into a combined folder
        Set paths in main()
        Run the script
'''

# Function to blur images
def blur_images(input_dir, output_dir):
    """
    Function blurs each image with a random degree of blurriness from 1-5 (can go up to 10).
    It also copies the corresponding YOLO label file, renaming it to match the blurred image.

    Input:
        input_dir - path to images and labels that need to be blurred
        output_dir - path where blurred images and corresponding labels will be saved
    """
    counter = 0

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_dir, filename)

            # Open the image
            image = Image.open(image_path)

            # Apply a random degree of blur
            blur_radius = random.uniform(1, 5)  # You can adjust the range (1,10)
            blurred_image = image.filter(ImageFilter.GaussianBlur(blur_radius))

            # Save the new image with '_blurry' added to the filename
            base, ext = os.path.splitext(os.path.basename(image_path))
            new_filename = f"{base}-blurry{ext}"
            blurred_image.save(os.path.join(output_dir, new_filename))

            # Find the corresponding YOLO label file and copy it with the new name
            label_file = f"{base}.txt"
            label_path = os.path.join(input_dir, label_file)
            new_label_filename = f"{base}-blurry.txt"
            new_label_path = os.path.join(output_dir, new_label_filename)

            if os.path.exists(label_path):
                shutil.copy(label_path, new_label_path)

            else:
                print(f"Warning: Label file {label_file} not found for image {filename}")

            counter += 1
            print(f"Processed {counter}: {new_filename} and {new_label_filename} (blurred and label copied)")

# Function to rotate images and adjust YOLO labels
def rotate_image_and_labels(input_dir, output_dir):
    """
    Rotates each image by a given angle using Albumentations and adjusts YOLO-style labels accordingly.

    Input:
        - input_dir: Directory containing images and YOLO labels
        - output_dir: Directory where rotated images and adjusted labels will be saved
        - rotation_angle: Angle to rotate (90, 180, or 270 degrees)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the Albumentations transform for rotation (using YOLO format)
    transform = A.Compose([
        A.Rotate(limit=(-180,180), p=1.0, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    counter = 0

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            label_path = os.path.join(input_dir, filename.replace('.png', '.txt'))

            # Open the image
            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]

            # Read the YOLO label file
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                bboxes = []
                class_labels = []
                for line in lines:
                    class_id, center_x, center_y, width, height = map(float, line.strip().split())
                    bboxes.append([center_x, center_y, width, height])  # YOLO format bounding boxes
                    class_labels.append(class_id)

                # Apply the transformation (rotation) to the image and bounding boxes
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']

                # Save the rotated image
                new_image_filename = filename.replace('.png', f'-rotated.png')
                cv2.imwrite(os.path.join(output_dir, new_image_filename), transformed_image)

                # Save the new YOLO label file
                new_label_filename = filename.replace('.png', f'-rotated.txt')
                with open(os.path.join(output_dir, new_label_filename), 'w') as f:
                    for bbox, class_id in zip(transformed_bboxes, class_labels):
                        f.write(f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
            else:
                print(f"Warning: Label file not found for {filename}")

            counter += 1
            print(f"Processed {counter}: {new_image_filename} and {new_label_filename} (rotation and label calculated)")



# Function to create train, val, test folders
def create_folders(base_output_directory):
    # Create train, val, test folders and their subfolders if they don't exist
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_output_directory, split)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

# Function to split data into train, val, test sets
def split_data(combined_directory, output_directory, augment=True, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Augment the images with blur and rotation if needed
    print("Running Script")
    if augment:
        print("Augmenting images with blurring...")
        blur_images(combined_directory, combined_directory)
        print("Rotating images...")
        rotate_image_and_labels(combined_directory, combined_directory)

    # Get all the image files in the directory
    image_files = [f for f in os.listdir(combined_directory) if f.endswith('.png')]
    print(f"Found {len(image_files)} images")

    # Shuffle the files
    random.shuffle(image_files)

    # Split the data
    total_images = len(image_files)
    train_count = floor(total_images * train_ratio)
    val_count = floor(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    return train_files, val_files, test_files

# Function to move files into the respective directories (train/val/test)
def move_files(files, combined_directory, target_directory):
    for file in files:
        # Move the image
        image_source = os.path.join(combined_directory, file)
        image_target = os.path.join(target_directory, 'images', file)
        shutil.move(image_source, image_target)

        # Move the corresponding label file from the same directory
        label_file = file.replace('.png', '.txt')
        label_source = os.path.join(combined_directory, label_file)
        if os.path.exists(label_source):
            label_target = os.path.join(target_directory, 'labels', label_file)
            shutil.move(label_source, label_target)
        else:
            print(f"Label file {label_file} does not exist for image {file}")

def main():
    # Input: Directory containing both images and labels
    combined_directory = 'combined_data'

    # Input: Where to create the output train, val, and test folders
    output_directory = '.'

    # Create train, val, test folders with subfolders in the output directory (if they don't exist)
    create_folders(output_directory)

    # Split the data and augment (blurring and rotating)
    train_files, val_files, test_files = split_data(
        combined_directory=combined_directory,
        output_directory=output_directory,
        augment=True,  # Set to True to perform blurring and rotation
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )

    # Move the files to their corresponding directories (train, val, test)
    move_files(train_files, combined_directory, os.path.join(output_directory, 'train'))
    move_files(val_files, combined_directory, os.path.join(output_directory, 'val'))
    move_files(test_files, combined_directory, os.path.join(output_directory, 'test'))

    print(f"Data split and move complete. Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    main()

