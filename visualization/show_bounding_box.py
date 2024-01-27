import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os


def plot_sample_images_with_bboxes(images_dir, labels_dir, sample_count=5):
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

    # Set up the plot
    fig, axs = plt.subplots(1, sample_count, figsize=(20, 5))

    # Loop through each sampled image and its corresponding label
    for i, (image_path, label_path) in enumerate(samples):
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load bounding box coordinates from the label file
        with open(label_path, 'r') as file:
            bbox_data = file.read().strip().split()

        # YOLO format: class, x_center, y_center, width, height
        x_center, y_center, width, height = map(float, bbox_data[1:])
        x_center *= image.width
        y_center *= image.height
        width *= image.width
        height *= image.height

        # Convert to matplotlib rectangle format: (x1, y1, width, height)
        rect = patches.Rectangle((x_center - width / 2, y_center - height / 2), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Plot image and add the bounding box
        axs[i].imshow(image)
        axs[i].add_patch(rect)
        axs[i].set_title(os.path.basename(image_path))
        axs[i].axis('off')

    # Display the plot
    plt.tight_layout()
    plt.show()


# Set your train and val directories paths
train_images_dir = 'datasets/images/train'
train_labels_dir = 'datasets/image_labels/train'
val_images_dir = 'datasets/images/val'
val_labels_dir = 'datasets/image_labels/val'

# Visualize training samples
print("Training Samples:")
plot_sample_images_with_bboxes(train_images_dir, train_labels_dir)

# Visualize validation samples
print("Validation Samples:")
plot_sample_images_with_bboxes(val_images_dir, val_labels_dir)
