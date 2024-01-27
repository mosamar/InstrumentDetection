import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def read_label_file(label_file_path, image_width, image_height):
    with open(label_file_path, 'r') as file:
        lines = file.readlines()

    objects = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coordinates = np.array([float(coord) for coord in parts[1:]], dtype=np.float32)
        coordinates = coordinates.reshape(-1, 2)  # Reshape to (num_points, 2)
        objects.append((class_id, coordinates))

    return objects


def create_mask(objects, image_width, image_height):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for class_id, coordinates in objects:
        # Scale coordinates back to image dimensions
        coordinates_scaled = np.array([(x * image_width, y * image_height) for x, y in coordinates], dtype=np.int32)
        cv2.fillPoly(mask, [coordinates_scaled], 255)  # Fill the polygon
    return mask


def visualize_segmentation(image_path, label_file_path):
    # Read original image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Read label file and create mask
    objects = read_label_file(label_file_path, image_width, image_height)
    mask = create_mask(objects, image_width, image_height)

    # Create an overlay with the desired mask color (yellow)
    overlay = image.copy()
    overlay_color = (0, 255, 255)  # Yellow color in BGR format
    overlay[mask == 255] = overlay_color

    # Blend the overlay with the original image using a transparency factor
    alpha = 0.3  # Transparency factor (between 0 and 1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying in matplotlib
    plt.axis('off')
    plt.show()


def show_random_samples(images_dir, labels_dir, sample_count=5):
    # Get the list of image files and corresponding label files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    label_files = [f.replace('.png', '.txt') for f in image_files]

    # Construct full file paths
    image_paths = [os.path.join(images_dir, file) for file in image_files]
    label_paths = [os.path.join(labels_dir, file) for file in label_files]

    # Make sure the sample_count is not greater than the dataset size
    sample_count = min(sample_count, len(image_paths))

    # Randomly select a few sample images and their corresponding image_labels
    samples = random.sample(list(zip(image_paths, label_paths)), sample_count)

    # Loop through each sampled image and its corresponding label
    for i, (image_path, label_path) in enumerate(samples):
        visualize_segmentation(image_path, label_path)


def plot_random_samples():
    train_images_dir = './datasets/images/train'
    train_labels_dir = './datasets/labels/train'
    val_images_dir = './datasets/images/val'
    val_labels_dir = './datasets/labels/val'

    # train_images_dir = '../data/guide_wire/images'
    # train_labels_dir = '../data/guide_wire/labels'
    # val_images_dir = './datasets/images/val'
    # val_labels_dir = './datasets/labels/val'
    # Visualize training samples
    print("Training Samples:")
    show_random_samples(train_images_dir, train_labels_dir)

    # Visualize validation samples
    print("Validation Samples:")
    show_random_samples(val_images_dir, val_labels_dir)


if __name__ == '__main__':
    # Example usage
    # visualize_segmentation('images/img-00016-00001_wire3.png',
    #                        'image_labels/img-00016-00001_wire3.txt')

    plot_random_samples()

    # visualize_segmentation('../vessel_model/datasets/images/1482969_Series_001_Frame43.png',
    #                        '../vessel_model/datasets/image_labels/1482969_Series_001_Frame43.txt')
