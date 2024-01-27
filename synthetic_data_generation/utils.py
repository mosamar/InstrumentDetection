import glob
import os
import random
import shutil

import cv2
import numpy as np
from PIL import ImageEnhance, ImageFilter, Image
from sklearn.model_selection import train_test_split


def apply_transformations(image):
    # Apply random rotation
    angle = random.randint(-180, 180)
    rotated_image = image.rotate(angle)

    # Apply random brightness change
    enhancer = ImageEnhance.Brightness(rotated_image)
    brightness_factor = random.uniform(0.5, 1.5)
    brightened_image = enhancer.enhance(brightness_factor)

    # Apply random contrast change
    contrast_enhancer = ImageEnhance.Contrast(brightened_image)
    contrast_factor = random.uniform(0.5, 1.5)
    final_image = contrast_enhancer.enhance(contrast_factor)

    return final_image


def add_noise_and_blur(img):
    # Generate a random kernel size for Gaussian blur (must be odd)
    kernel_size = random.choice([3, 5, 7, 9])

    # Convert to OpenCV format
    open_cv_image = np.array(img)

    # Apply Gaussian blur
    blur_img = cv2.GaussianBlur(open_cv_image, (kernel_size, kernel_size), 0)

    # Check if the image is already grayscale
    if len(blur_img.shape) == 2 or blur_img.shape[2] == 1:
        gray_img = blur_img  # Image is already grayscale
    else:
        # Convert to grayscale if it's a color image
        gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)

    # Generate random Gaussian noise
    mean = 10
    stddev = random.uniform(0, 30)
    noise = np.zeros(gray_img.shape, np.uint8)
    cv2.randn(noise, mean, stddev)

    # Add noise to image
    noisy_img = cv2.add(gray_img, noise)

    # Convert back to PIL format
    pil_image = Image.fromarray(noisy_img)

    return pil_image

    # Add noise to image
    noisy_img = cv2.add(gray_img, noise)

    # Convert back to PIL format
    pil_image = Image.fromarray(noisy_img)

    return pil_image


def move_files(files, source_dir, target_dir):
    for file in files:
        os.rename(os.path.join(source_dir, file), os.path.join(target_dir, file))


def copy_files(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Copy each file from the source directory to the target directory
    for file_path in glob.glob(os.path.join(source_dir, '*')):
        shutil.copy(file_path, target_dir)


def arrange():
    output_dir = os.path.join('datasets', "images")  # Directory where synthetic images will be saved
    label_dir = os.path.join('datasets', "labels")  # Directory where label files will be saved
    #
    #
    # Make sure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Define paths for train and validation splits
    train_images_dir = os.path.join(output_dir, 'train')
    val_images_dir = os.path.join(output_dir, 'val')
    train_labels_dir = os.path.join(label_dir, 'train')
    val_labels_dir = os.path.join(label_dir, 'val')

    # Create directories for train and val splits if they don't exist
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Get list of generated image and label files
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    label_files = [f.replace('.png', '.txt') for f in image_files]

    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=0.2, random_state=42)

    # Move the files to their respective train/val directories
    move_files(train_images, output_dir, train_images_dir)
    move_files(val_images, output_dir, val_images_dir)
    move_files(train_labels, label_dir, train_labels_dir)
    move_files(val_labels, label_dir, val_labels_dir)
