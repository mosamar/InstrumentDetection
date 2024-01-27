import glob

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO('runs/segment/train/weights/best.pt').to(device=0)


def create_segmentation_mask(img_np, mask_coordinates, classes):
    class_colors = {
        0: [0, 0, 255, 128]  # Blue for class 3 (with alpha value)
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
    results = model(img, save=True, imgsz=640, conf=0.1, half=False)
    img_np = np.array(img)
    for r in results:
        image_height, image_width = img_np.shape[:2]

        mask_coord = r.masks.xy
        classes = r.boxes.cls.cpu().numpy()
        # boxes = r.boxes
        # print(boxes)

        img_np = create_segmentation_mask(img_np, mask_coord, classes)

    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_cv


def detect_vessels(image_path):
    for image_file in glob.glob(image_path + '/*.png'):
        img = Image.open(image_file)
        img_cv = run_model(img)
        cv2.imwrite('runs/detect/trace_vessels/' + image_file, img_cv)
        cv2.namedWindow('YOLO V8 Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO V8 Detection', img_cv)
        if cv2.waitKey(0):
            continue


if __name__ == '__main__':
    image_path = 'datasets/test/'
    detect_vessels(image_path)
    cv2.destroyAllWindows()
