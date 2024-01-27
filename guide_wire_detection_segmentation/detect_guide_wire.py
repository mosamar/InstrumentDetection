import glob
import os.path
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

model = YOLO('runs/segment/train/weights/best.pt').to(device=0)


def create_segmentation_mask(img_np, mask_coordinates, classes):
    class_colors = {
        0: [0, 0, 255, 128],  # Example color for class 0 (with alpha value)
        1: [0, 255, 0, 128],  # Green for class 1 (with alpha value)
        2: [255, 0, 0, 128],  # Red for class 2 (with alpha value)
        3: [0, 0, 255, 128]  # Blue for class 3 (with alpha value)
    }

    # Ensure the image has 4 channels (RGBA)
    if img_np.shape[2] == 3:
        img_np = np.concatenate((img_np, np.full(img_np.shape[:2] + (1,), 255, dtype=np.uint8)), axis=2)

    mask_overlay = np.zeros_like(img_np)

    for coords, cls in zip(mask_coordinates, classes):
        color = class_colors[int(cls)]
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)

        # Convert coordinates to a format suitable for cv2.fillPoly
        contour = np.array(coords).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [contour], color=[1, 1, 1])  # Fill contour with white

        # Overlay mask onto the mask_overlay with the class-specific color
        mask_overlay[mask == 1] = color

    # Combine original image with the mask overlay
    img_with_mask = cv2.addWeighted(img_np, 1, mask_overlay, 0.5, 0)

    return img_with_mask


def run_model(img):
    results = model(img, save=True, imgsz=640, conf=0.5, half=False)
    img_np = np.array(img)
    image_height, image_width = img_np.shape[:2]
    for r in results:
        if r.masks:
            mask_coord = r.masks.xy
            classes = r.boxes.cls.cpu().numpy()
            img_np = create_segmentation_mask(img_np, mask_coord, classes)

        boxes = r.boxes
        # print(boxes)

    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_cv


def detect_wire(image_folder):
    if os.path.isdir(image_folder):
        for image_file in glob.glob(image_folder + '/*.png'):
            img = Image.open(image_file)
            img_cv = run_model(img)
            cv2.imwrite('runs/detect/trace_wire/' + image_file, img_cv)
            cv2.namedWindow('YOLO V8 Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('YOLO V8 Detection', img_cv)
            if cv2.waitKey(0):
                continue
    elif os.path.isfile(image_folder):
        img = Image.open(image_folder)
        img_cv = run_model(img)
        cv2.imwrite('runs/detect/trace_wire/' + image_folder, img_cv)
        cv2.namedWindow('YOLO V8 Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO V8 Detection', img_cv)


if __name__ == '__main__':
    image_path = 'datasets/test/'
    detect_wire(image_path)
    cv2.destroyAllWindows()

    # cap = cv2.VideoCapture('Animalstudy_run_afterpreprocessing.avi')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('Animalstudy_run_afterpreprocessing_output.avi', fourcc, 20.0, (640, 480))
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         try:
    #             processed_frame = run_model(frame)
    #             out.write(processed_frame)
    #         except Exception as e:
    #             print(e)
    #     else:
    #         break
    #
    # cap.release()
    # out.release()
