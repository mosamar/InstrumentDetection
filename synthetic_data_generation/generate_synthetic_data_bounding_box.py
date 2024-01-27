import glob

from PIL import Image, ImageDraw
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split


def overlay_screw_and_label(xray_path, screw_paths, output_dir, label_dir):
    print('Processing', xray_path)
    base_name = os.path.splitext(os.path.basename(xray_path))[0]
    xray = Image.open(xray_path).convert("RGBA")
    xray_width, xray_height = xray.size

    # Overlay multiple screws
    label_str = ""
    for screw_path in screw_paths:
        screw_name = os.path.splitext(os.path.basename(screw_path))[0]
        base_name = base_name + '_' + screw_name

        screw = Image.open(screw_path).convert("RGBA")
        scale = random.uniform(0.7, 1)
        angle = random.randint(0, 360)
        screw = screw.rotate(angle, expand=True)
        screw_size = tuple([int(dim * scale) for dim in screw.size])
        screw = screw.resize(screw_size, Image.Resampling.LANCZOS)

        # Adjust for partial entry
        x_pos = random.randint(-screw_size[0] // 2, xray_width - screw_size[0] // 2)
        y_pos = random.randint(-screw_size[1] // 2, xray_height - screw_size[1] // 2)
        xray.paste(screw, (x_pos, y_pos), screw)

        # Calculate bounding box
        bbox = [max(0, x_pos), max(0, y_pos), min(xray_width, x_pos + screw_size[0]),
                min(xray_height, y_pos + screw_size[1])]
        x_center = ((bbox[0] + bbox[2]) / 2) / xray_width
        y_center = ((bbox[1] + bbox[3]) / 2) / xray_height
        norm_width = (bbox[2] - bbox[0]) / xray_width
        norm_height = (bbox[3] - bbox[1]) / xray_height

        screw_class = 0 if screw_name == "screw1" else 1 if screw_name == "screw2" else 2
        label_str += f"{screw_class} {x_center} {y_center} {norm_width} {norm_height}\n"

    output_image_path = os.path.join(output_dir, f"{base_name}_multiple_screws.png")
    label_file_path = os.path.join(label_dir, f"{base_name}_multiple_screws.txt")

    # Save the synthetic image and label
    xray.save(output_image_path)
    with open(label_file_path, 'w') as label_file:
        label_file.write(label_str)


def generate_bbox_data(data_dir, dataset_dir):
    xray_dir = os.path.join(data_dir, "original_data")  # Directory containing original X-ray images
    screw_dir = os.path.join(data_dir, "screws")
    output_dir = os.path.join(dataset_dir, "images")  # Directory where synthetic images will be saved
    label_dir = os.path.join(dataset_dir, "image_labels")
    # Process each X-ray image to create synthetic data
    for xray_file in os.listdir(xray_dir):
        xray_path = os.path.join(xray_dir, xray_file)
        if os.path.isfile(xray_path):
            screw_images = glob.glob(screw_dir + "/*.png")
            for i in range(len(screw_images)):
                screw_subset = screw_images[:i + 1]
                overlay_screw_and_label(xray_path, screw_subset, output_dir, label_dir)
