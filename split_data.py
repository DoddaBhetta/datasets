import os
import random
import shutil
from math import floor

'''
    This script automates the process of splitting image and label data into training, 
    validation, and test sets. It organizes the images and their corresponding labels into 
    separate directories (train, val, and test), ensuring that both image files (.png) 
    and their matching label files (.txt) are moved together. The data is split by a ratio 
    of 80% for training, 10% for validation, and 10% for testing, and new folders are created 
    for each of these splits.

    You would run this script whenever you get new data, ensuring that the data is correctly organized 
    and split for model training and evaluation.

'''


def create_folders(base_output_directory):
    # Create train, val, test folders and their subfolders
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_output_directory, split)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)


def split_data(image_directory, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Get all the image files in the directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith('.png')]

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


def move_files(files, image_directory, label_directory, target_directory):
    for file in files:
        # Move the image
        image_source = os.path.join(image_directory, file)
        image_target = os.path.join(target_directory, 'images', file)
        shutil.copy(image_source, image_target)

        # Move the corresponding label file from the separate label directory
        label_file = file.replace('.png', '.txt')
        label_source = os.path.join(label_directory, label_file)
        if os.path.exists(label_source):
            label_target = os.path.join(target_directory, 'labels', label_file)
            shutil.copy(label_source, label_target)
        else:
            print(f"Warning: Label file {label_file} does not exist for image {file}")


def main():
    # Input: Directory containing images
    image_directory = '/Users/elmorajahm1/Desktop/datasets/combined_data/images'

    # Input: Directory containing labels
    label_directory = '/Users/elmorajahm1/Desktop/datasets/combined_data/labels'

    # Input: Where to create the output train, val, and test folders
    output_directory = '/Users/elmorajahm1/Desktop/datasets'

    # Create train, val, test folders with subfolders in the output directory
    create_folders(output_directory)

    # Split the data
    train_files, val_files, test_files = split_data(image_directory)

    # Move the files to their corresponding directories (train, val, test)
    move_files(train_files, image_directory, label_directory, os.path.join(output_directory, 'train'))
    move_files(val_files, image_directory, label_directory, os.path.join(output_directory, 'val'))
    move_files(test_files, image_directory, label_directory, os.path.join(output_directory, 'test'))

    print(f"Data split complete. Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


if __name__ == "__main__":
    main()
