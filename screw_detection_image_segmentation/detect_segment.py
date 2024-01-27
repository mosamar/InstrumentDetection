import glob
import random

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO

model = YOLO('runs/segment/train/weights/best.pt').to(device=0)
x_crosshair = 500
y_crosshair = 512


def extend_line(image_width, image_height, x1, y1, x2, y2):
    """Extend a line from a point through another point to the image edge."""
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)  # Slope of the line
    else:
        # Line is vertical
        return [(0, y1), (image_width, y1)]

    b = y1 - m * x1  # Y-intercept

    # Calculate intersection points with the image edges
    y_at_x0 = b
    y_at_xmax = m * image_width + b
    x_at_y0 = -b / m if m != 0 else 0
    x_at_ymax = (image_height - b) / m if m != 0 else image_width

    # Determine which points are within the image boundaries
    points = [(0, int(y_at_x0)), (image_width, int(y_at_xmax)), (int(x_at_y0), 0), (int(x_at_ymax), image_height)]
    points = [p for p in points if 0 <= p[0] <= image_width and 0 <= p[1] <= image_height]

    # Choose two points to form the extended line
    return sorted(points, key=lambda p: (p[0] - x1) ** 2 + (p[1] - y1) ** 2)[:2]


def find_longest_line(cropped_img):
    # Check if the image is already grayscale
    if len(cropped_img.shape) == 2:
        gray = cropped_img
    else:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)  # Adjust these values as needed

    # Use Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        # Find the longest line
        longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][:2] - line[0][2:4]))
        return longest_line[0]
    return None


def draw_parallel_line(image, original_line, midpoint):
    """Draw a line parallel to original_line passing through midpoint."""
    x1, y1, x2, y2 = original_line
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)  # Slope of the original line
        # Calculate y-intercept of the parallel line
        b_parallel = midpoint[1] - m * midpoint[0]
        # Define points for the parallel line
        y_at_x0_parallel = b_parallel
        y_at_xmax_parallel = m * image.shape[1] + b_parallel
        return [(0, int(y_at_x0_parallel)), (image.shape[1], int(y_at_xmax_parallel))]
    else:
        # If the line is vertical, draw a vertical line through the midpoint
        return [(midpoint[0], 0), (midpoint[0], image.shape[0])]


# Load the YOLO model
def detect_screw(image_path):
    for image_file in glob.glob(image_path + '/*.png'):
        img = Image.open(image_file)
        img_cv = run_model(img)
        img_cv = cv2.convertScaleAbs(img_cv)
        cv2.imwrite('runs/detect/trace_screw/' + image_file, img_cv)
        cv2.namedWindow('YOLO V8 Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO V8 Detection', img_cv)
        if cv2.waitKey(0):
            continue


def run_model(img):
    results = model(img, save=True, imgsz=1024, conf=0.6, half=False)

    for r in results:
        img_np = np.array(img)
        image_height, image_width = img_np.shape[:2]

        for box in r.boxes:
            b = box.xyxy[0]
            bbox = [int(val) for val in b]

            # Validate bbox coordinates
            bbox[0] = max(0, min(bbox[0], image_width - 1))
            bbox[1] = max(0, min(bbox[1], image_height - 1))
            bbox[2] = max(0, min(bbox[2], image_width - 1))
            bbox[3] = max(0, min(bbox[3], image_height - 1))

            cropped_img = img_np[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            b = box.xyxy[0]
            bbox = [int(val) for val in b]
            # Calculate the center of the bounding box
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2

            # Check if the cropped image is valid
            if cropped_img.size == 0 or len(cropped_img.shape) < 2:
                continue  # Skip if the cropped image is invalid

            line = find_longest_line(cropped_img)
            if line is not None:
                adjusted_line = (line[0] + bbox[0], line[1] + bbox[1], line[2] + bbox[0], line[3] + bbox[1])

                extended_line = extend_line(image_width, image_height, *adjusted_line)
                # cv2.line(img_np, extended_line[0], extended_line[1], (0, 255, 0), 2)

                # Calculate midpoint of the screw's width
                screw_midpoint = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2

                # Draw a parallel line through the screw's midpoint
                parallel_line = draw_parallel_line(img_np, adjusted_line, screw_midpoint)
                drawline(img_np, parallel_line[0], parallel_line[1], (255, 0, 0),
                         2, gap=10)  # Red line for the new parallel line

                img_cv = add_random_point(img_np, x_crosshair, y_crosshair)

                img_cv = drawline(img_cv, (x_crosshair, y_crosshair), (int(x_center), int(y_center)),
                                  (0, 255, 0), 2, gap=10)

                # Calculate the angle
                angle = angle_between_lines(extended_line, ((x_crosshair, y_crosshair), (int(x_center), int(y_center))))

                # Position for the angle label (you can adjust this as needed)
                label_position = ((x_crosshair + int(x_center)) // 2, (y_crosshair + int(y_center)) // 2)

                # Draw the angle
                img_np = draw_angle_label(img_np, angle, label_position)

        img_cv = img_np  # cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_cv


def draw_angle_label(img, angle, position):
    """Draw the angle label on the image."""
    font_path = 'C:\\Windows\\Fonts\\Arial.ttf'
    font_size = 48
    font = ImageFont.truetype(font_path, font_size)
    font_color = (0, 0, 255)  # White color
    text = f"{angle:.1f}°"  # Format angle to 2 decimal places

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text=text, font=font, fill=255)
    # Convert back to OpenCV image format
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_cv


def calculate_slope(point1, point2):
    """Calculate the slope of a line segment."""
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:  # Avoid division by zero
        return np.inf  # Infinite slope for vertical lines
    return (y2 - y1) / (x2 - x1)


def angle_between_lines(line1, line2):
    """Calculate the angle between two lines."""
    # Unpack points from the lines
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # Calculate slopes
    m1 = calculate_slope((x1, y1), (x2, y2))
    m2 = calculate_slope((x3, y3), (x4, y4))

    # Calculate angle in radians
    angle_rad = np.arctan(abs((m2 - m1) / (1 + m1 * m2)))

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1
    return img


def add_random_point(img, x, y):
    # Get image dimensions
    height, width = img.shape[:2]

    # Define crosshair properties
    crosshair_size = 20  # Length of the crosshair arms
    crosshair_color = (0, 255, 0)  # Green color
    crosshair_thickness = 2  # Thickness of the crosshair lines

    # Draw the crosshair
    cv2.line(img, (x - crosshair_size, y), (x + crosshair_size, y), crosshair_color, crosshair_thickness)
    marked_img = cv2.line(img, (x, y - crosshair_size), (x, y + crosshair_size), crosshair_color, crosshair_thickness)
    return marked_img


if __name__ == '__main__':
    image_path = 'datasets/test/'
    detect_screw(image_path)
    cv2.destroyAllWindows()
