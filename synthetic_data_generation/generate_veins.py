import shutil

import cv2
import numpy as np
import random
import os

import cv2
import numpy as np
import random
import os
import glob
from itertools import combinations

from tqdm import tqdm

from synthetic_data_generation.utils import copy_files, arrange


def overlay_vessels_randomly(base_image_path, vessel_images, output_image_path, output_label_path, class_id):
    base_image = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)

    all_contours = []

    for vessel_image_path in vessel_images:
        vessel_image = cv2.imread(vessel_image_path, cv2.IMREAD_UNCHANGED)

        # Apply random transformations
        scale = random.uniform(0.5, 1.5)
        angle = random.randint(0, 360)
        rotation_matrix = cv2.getRotationMatrix2D((vessel_image.shape[1] / 2, vessel_image.shape[0] / 2), angle, scale)
        vessel_image = cv2.warpAffine(vessel_image, rotation_matrix, (vessel_image.shape[1], vessel_image.shape[0]))

        # Random location
        max_x = base_image.shape[1] - vessel_image.shape[1]
        max_y = base_image.shape[0] - vessel_image.shape[0]
        top_left_x = random.randint(0, max_x)
        top_left_y = random.randint(0, max_y)

        # Overlay
        alpha_vessel = vessel_image[:, :, 3] / 255.0
        alpha_base = 1.0 - alpha_vessel
        for c in range(0, 3):
            base_image[top_left_y:top_left_y + vessel_image.shape[0], top_left_x:top_left_x + vessel_image.shape[1],
            c] = \
                (alpha_vessel * vessel_image[:, :, c] + alpha_base * base_image[
                                                                     top_left_y:top_left_y + vessel_image.shape[0],
                                                                     top_left_x:top_left_x + vessel_image.shape[1], c])

        # Calculate and store contours for YOLO v8 format
        contours, _ = cv2.findContours(vessel_image[:, :, 3], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            adjusted_contour = contour + [top_left_x, top_left_y]
            all_contours.append(adjusted_contour)

    # Save the result
    cv2.imwrite(output_image_path, base_image)

    # Write image_labels
    with open(output_label_path, 'w') as f:
        for contour in all_contours:
            f.write(f'{class_id}')
            for point in contour.squeeze():
                norm_x, norm_y = point[0] / base_image.shape[1], point[1] / base_image.shape[0]
                f.write(f' {norm_x} {norm_y}')
            f.write('\n')


def generate_veins_data(base_img_path, vessel_img_dir, img_output_dir, label_output_dir, num_variations=3):
    vessel_images = glob.glob(os.path.join(vessel_img_dir, '*.png'))

    base_name = os.path.splitext(os.path.basename(base_img_path))[0]

    for n_vessels in range(1, 3):  # 1 to 3 vessels
        for variation in range(1, num_variations + 1):  # Multiple variations
            selected_vessels = random.sample(vessel_images, n_vessels)
            output_image_path = os.path.join(img_output_dir, f'{base_name}_var{variation}_{n_vessels}vessels.png')
            output_label_path = os.path.join(label_output_dir, f'{base_name}_var{variation}_{n_vessels}vessels.txt')

            overlay_vessels_randomly(base_img_path, selected_vessels, output_image_path, output_label_path, 4)


def generate_training_data(data_dir, dataset_dir, generate_augmented_data=False):
    xray_dir = os.path.join(data_dir, "original_data")  # Directory containing original X-ray images
    vessels_dir = os.path.join(data_dir, "updated_vessels")
    output_dir = os.path.join(dataset_dir, "images")  # Directory where synthetic images will be saved
    label_dir = os.path.join(dataset_dir, "labels")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    copy_files('../data/vessel_image_data/lightned_images', output_dir)
    copy_files('../data/vessel_image_data/labels', label_dir)

    if generate_augmented_data:
        for xray_file in tqdm(os.listdir(xray_dir), desc="Processing X-ray images"):
            xray_path = os.path.join(xray_dir, xray_file)
            generate_veins_data(xray_path, vessels_dir, output_dir, label_dir)

# if __name__ == '__main__':
#     # Example usage
#     vessel_image_dir = os.path.join(data_dir, "vessel_image_data/image_data/updated_vessels")
#
#     generate_training_data()
#     arrange()
