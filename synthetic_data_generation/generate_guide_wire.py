import random
from math import comb

import numpy as np
from PIL import Image, ImageDraw

# from utils import add_noise_and_blur
from visualization.visualize_segmentation import visualize_segmentation


def bernstein_poly(i, n, t):
    """
    Compute the Bernstein polynomial of n, i as a function of t.
    """
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))


def bezier_curve(points, num_points=1000):
    """
    Generate Bezier curve from control points.

    Args:
    points (list of tuples): Control points.
    num_points (int): Number of points to generate on the curve.

    Returns:
    list of tuples: Points on the Bezier curve.
    """
    n_points = len(points)
    x_vals, y_vals = np.array([0.0] * num_points), np.array([0.0] * num_points)

    for i in range(n_points):
        bernstein = bernstein_poly(i, n_points - 1, np.linspace(0.0, 1.0, num_points))
        x_vals += points[i][0] * bernstein
        y_vals += points[i][1] * bernstein

    return x_vals, y_vals


def generate_random_bezier_curve(img_size=(1000, 1000)):
    """
    Generate a random Bezier curve starting from an edge of the image.

    Args:
    num_points (int): Number of control points.
    img_size (tuple): Size of the image (width, height).

    Returns:
    tuple: bezier_x, bezier_y - Coordinates of the Bezier curve.
    """
    num_points = random.randint(3, 6)
    # Randomly select an edge for the starting point (0: left, 1: right, 2: top, 3: bottom)
    edge = np.random.choice(['left', 'right', 'top', 'bottom'])

    # Determine the starting point on the chosen edge
    if edge == 'left':
        start_point = (0, np.random.uniform(0, img_size[1]))
    elif edge == 'right':
        start_point = (img_size[0], np.random.uniform(0, img_size[1]))
    elif edge == 'top':
        start_point = (np.random.uniform(0, img_size[0]), 0)
    else:  # 'bottom'
        start_point = (np.random.uniform(0, img_size[0]), img_size[1])

    # Generate the remaining control points randomly within the image
    points = [start_point] + [(np.random.uniform(0, img_size[0]), np.random.uniform(0, img_size[1]))
                              for _ in range(num_points - 1)]

    # Generate Bezier curve points
    bezier_x, bezier_y = bezier_curve(points)

    return bezier_x, bezier_y


def draw_curve_on_image(img, bezier_x, bezier_y, overlay_width=2, antialias_factor=10):
    """
    Draw the bezier curve on the image with a smoother and more natural appearance.

    Args:
    img (PIL.Image): Image on which to draw the curve.
    bezier_x (list): x-coordinates of the curve.
    bezier_y (list): y-coordinates of the curve.
    overlay_width (int): Width of the overlay line.
    antialias_factor (int): Factor for antialiasing, higher for smoother curves.

    Returns:
    PIL.Image: Image with the curve drawn in a more natural and smooth way.
    """
    # Create an image for drawing the curve with antialiasing
    aa_size = (img.width * antialias_factor, img.height * antialias_factor)
    aa_img = Image.new("RGBA", aa_size, (255, 255, 255, 20))

    # Adjust the bezier curve points and width for the antialiased image
    aa_bezier_x = [x * antialias_factor for x in bezier_x]
    aa_bezier_y = [y * antialias_factor for y in bezier_y]
    aa_overlay_width = overlay_width * antialias_factor

    # Draw the antialiased curve
    draw = ImageDraw.Draw(aa_img, "RGBA")
    aa_points = list(zip(aa_bezier_x, aa_bezier_y))
    curve_color = (0, 0, 0, 10)  # Semi-transparent black
    draw.line(aa_points, fill=curve_color, width=aa_overlay_width)

    # Resize the curve image back to the original size for smooth curve
    aa_img = aa_img.resize(img.size, Image.Resampling.LANCZOS)

    # Blend the antialiased curve image with the original image
    blended_img = Image.alpha_composite(img.convert("RGBA"), aa_img)

    return blended_img


def create_yolo_v8_labels(class_id, bezier_x, bezier_y, img_size=(1000, 1000), overlay_width=30):
    """
    Create YOLO v8 image_labels for the overlay of the curve as a polygon.

    Args:
    class_id (int): The class ID for the curve.
    bezier_x (list): x-coordinates of the curve.
    bezier_y (list): y-coordinates of the curve.
    img_size (tuple): Size of the image (width, height).
    overlay_width (int): Width of the overlay line.

    Returns:
    str: Formatted label string for YOLO v8.
    """
    half_width_normalized = overlay_width / 2 / img_size[1]  # Normalizing based on image height
    polygon_coords = []

    # Create the top and bottom lines of the overlay polygon
    for i in range(len(bezier_x)):
        x, y = bezier_x[i] / img_size[0], bezier_y[i] / img_size[1]  # Normalize x and y
        polygon_coords.append(
            (x, min(max(y + half_width_normalized, 0.0), 1.0)))  # Ensure y + half_width is within [0, 1]
    for i in reversed(range(len(bezier_x))):
        x, y = bezier_x[i] / img_size[0], bezier_y[i] / img_size[1]  # Normalize x and y
        polygon_coords.append(
            (x, min(max(y - half_width_normalized, 0.0), 1.0)))  # Ensure y - half_width is within [0, 1]

    # Flatten the list and format as strings
    flattened_coords = ' '.join([f'{coord:.6f}' for xy in polygon_coords for coord in xy])

    # Format the coordinates as a YOLO v8 polygon label
    label = f"{class_id} {flattened_coords}"
    return label


# Rest of your code remains the same


# Rest of your code remains the same


# Generate and overlay the guidewire
def generate_xray_with_guide_wire(img_path):
    img = Image.open(img_path)
    bezier_x, bezier_y = generate_random_bezier_curve(img_size=img.size)
    xray_with_curve = draw_curve_on_image(img, bezier_x, bezier_y, overlay_width=2)
    # Create YOLO v8 segmentation label
    yolo_label = create_yolo_v8_labels(0, bezier_x, bezier_y, img_size=img.size)
    # Save the image and the label
    xray_with_curve.save("xray_with_guidewire.png")
    with open("xray_with_guidewire.txt", "w") as label_file:
        label_file.write(yolo_label)


# Display the image (optional)
# xray_with_curve.show()
if __name__ == '__main__':
    generate_xray_with_guide_wire('../data/original_data/img-00016-00022.png')
    visualize_segmentation('xray_with_guidewire.png',
                           'xray_with_guidewire.txt')
