import os.path
import traceback
from itertools import combinations
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from synthetic_data_generation.generate_guide_wire import generate_random_bezier_curve, draw_curve_on_image, \
    create_yolo_v8_labels
from synthetic_data_generation.utils import *
from visualization.visualize_segmentation import visualize_segmentation


def is_boundary_pixel(alpha_channel, x, y):
    """Check if the pixel is a boundary pixel"""
    rows, cols = alpha_channel.shape
    if alpha_channel[y, x] == 0:
        return False

    # Check neighbors (up, down, left, right)
    if y > 0 and alpha_channel[y - 1, x] == 0:
        return True
    if y < rows - 1 and alpha_channel[y + 1, x] == 0:
        return True
    if x > 0 and alpha_channel[y, x - 1] == 0:
        return True
    if x < cols - 1 and alpha_channel[y, x + 1] == 0:
        return True

    return False


def get_contour_coordinates(image, x_offset, y_offset, image_width, image_height):
    # Convert image to an array
    image_array = np.array(image)

    # Check if the image has an alpha channel
    if image_array.shape[2] == 4:
        alpha_channel = image_array[:, :, 3]
    else:
        raise ValueError("Image does not contain an alpha channel")

    # Create a binary mask where the alpha channel is not transparent
    binary_mask = np.uint8(alpha_channel > 0)

    # Find contours from the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Get the largest contour and approximate it to a polygon
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.0001 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True)

    # Normalize the coordinates of the polygon
    normalized_polygon = []
    for point in polygon:
        normalized_x = (x_offset + point[0][0]) / image_width
        normalized_y = (y_offset + point[0][1]) / image_height
        normalized_polygon.append((normalized_x, normalized_y))

    return normalized_polygon


def overlay_instrument_on_xray_image(xray_image, instrument_image, class_id, start_from_edge=False):
    xray_width, xray_height = xray_image.size
    # Apply random transformations to the instrument
    scale = random.uniform(0.5, 1.0)
    angle = random.randint(0, 360)
    instrument_image = instrument_image.rotate(angle, expand=True)
    instr_image_size = tuple([int(dim * scale) for dim in instrument_image.size])

    # Ensure the instrument image is not larger than the x-ray image
    instr_image_size = (min(instr_image_size[0], xray_width), min(instr_image_size[1], xray_height))
    instrument_image = instrument_image.resize(instr_image_size, Image.Resampling.LANCZOS)

    if start_from_edge:
        # Choose which edge to start from (0: top, 1: right, 2: bottom, 3: left)
        edge = random.randint(0, 3)

        if edge == 0:  # Top edge
            x_pos = random.randint(0, xray_width - instr_image_size[0])
            y_pos = 0
        elif edge == 1:  # Right edge
            x_pos = xray_width - instr_image_size[0]
            y_pos = random.randint(0, xray_height - instr_image_size[1])
        elif edge == 2:  # Bottom edge
            x_pos = random.randint(0, xray_width - instr_image_size[0])
            y_pos = xray_height - instr_image_size[1]
        else:  # Left edge
            x_pos = 0
            y_pos = random.randint(0, xray_height - instr_image_size[1])
    else:
        # Choose a random position within the x-ray image
        x_pos = random.randint(0, xray_width - instr_image_size[0])
        y_pos = random.randint(0, xray_height - instr_image_size[1])

    # Paste the screw onto the X-ray image
    xray_image.paste(instrument_image, (x_pos, y_pos), instrument_image)

    # Get the contour coordinates for this screw
    contour_coords = get_contour_coordinates(instrument_image, x_pos, y_pos, xray_width, xray_height)

    # Format the coordinates for YOLO v8 (as polygons)
    label_line = f"{class_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in contour_coords)

    return xray_image, label_line


def save_image_n_label(image, label_lines, base_name, instrument, image_dir, label_dir):
    image_name = f"{base_name}_{instrument}"
    output_image_path = os.path.join(image_dir, f"{image_name}.png")
    label_file_path = os.path.join(label_dir, f"{image_name}.txt")
    try:
        with open(output_image_path, 'wb') as f:
            image.save(f, format='PNG')
    except Exception as e:
        print(f"Failed to save image {output_image_path}: {e}")
        traceback.print_exc()

    # Write all segmentation coordinates to the label file
    with open(label_file_path, 'w') as label_file:
        label_file.write("\n".join(label_lines))


def overlay_screws_and_label(base_name, xray_path, selected_screws, image_dir, label_dir):
    # Overlay each screw
    for screw_paths in selected_screws:
        # Open the base X-ray image
        xray = Image.open(xray_path).convert("RGBA")
        xray = apply_transformations(xray)
        # xray_width, xray_height = xray.size
        label_lines = []
        # print(list(screw_paths))
        save_file_name = ''
        for screw_path in list(screw_paths):
            screw = Image.open(screw_path).convert("RGBA")
            screw_name = str(os.path.splitext(os.path.basename(screw_path))[0])
            save_file_name += screw_name
            instr_map = {'drill': 0, 'screw1': 1, 'screw2': 2,
                         'screw3': 3, 'screw4': 4, 't-screw': 5, 'k-wire': 6}
            class_id = instr_map[screw_name]

            xray, labels = overlay_instrument_on_xray_image(xray, screw, class_id)
            # Apply transformations
            xray_with_noise = add_noise_and_blur(xray)

            label_lines.append(labels)
            save_image_n_label(xray, label_lines, base_name, save_file_name, image_dir, label_dir)
            save_image_n_label(xray_with_noise, label_lines, base_name + '_with_noise', save_file_name, image_dir,
                               label_dir)


# Function to move files to the respective train/val directories

def generate_guide_wire_and_stent_data(base_name, xray_path, i, output_dir, label_dir, overlay_width, class_id):
    # Open and convert the X-ray image
    xray = Image.open(xray_path).convert("RGBA")

    # Apply transformations
    xray_transformed = apply_transformations(xray)

    xray_transformed_noisy = add_noise_and_blur(xray_transformed)

    # Generate a random Bezier curve
    bezier_x, bezier_y = generate_random_bezier_curve(img_size=xray.size)

    # Draw the curve on the transformed image
    xray_with_curve = draw_curve_on_image(xray_transformed_noisy, bezier_x, bezier_y, overlay_width=overlay_width)

    # Create YOLO v8 segmentation label
    yolo_label = create_yolo_v8_labels(class_id, bezier_x, bezier_y, img_size=xray.size)

    # Save the processed image and label
    save_image_n_label(xray_with_curve, [yolo_label], base_name, f'wire{i}', output_dir, label_dir)

    # Create negative data
    save_image_n_label(xray_transformed_noisy, [], base_name + '_negative', f'wire{i}', output_dir, label_dir)


def process_xray(args):
    xray_file, xray_dir, screw_paths, output_dir, label_dir, screw, wire, stent = args
    xray_path = os.path.join(xray_dir, xray_file)
    if os.path.isfile(xray_path):
        base_name = os.path.splitext(os.path.basename(xray_path))[0]
        for i in range(1, len(screw_paths) + 1):
            selected_screws = list(combinations(screw_paths, i))
            if screw:
                overlay_screws_and_label(base_name, xray_path, selected_screws, output_dir, label_dir)
            if wire:
                number_of_variations = 20
                overlay_width = 3
                for j in range(number_of_variations):
                    generate_guide_wire_and_stent_data(base_name, xray_path, j, output_dir, label_dir, overlay_width, 0)
            if stent:
                number_of_variations = 20
                overlay_width = 10
                for j in range(number_of_variations):
                    generate_guide_wire_and_stent_data(base_name, xray_path, j, output_dir, label_dir, overlay_width, 1)


def generate_training_data(data_dir, dataset_dir, screw=False, wire=False, stent=False):
    xray_dir = os.path.join(data_dir, "original_data")
    screw_dir = os.path.join(data_dir, "screws")
    output_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    screw_paths = sorted(glob.glob(os.path.join(screw_dir, "*.png")))
    xray_files = [(file, xray_dir, screw_paths, output_dir, label_dir, screw, wire, stent) for file in
                  os.listdir(xray_dir)]

    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(process_xray, xray_files), total=len(xray_files), desc="Processing X-ray images"))





if __name__ == '__main__':
    screw_paths = sorted(glob.glob(os.path.join('../data/screws', "*.png")))
    selected_screws = list(combinations(screw_paths, 1))
    overlay_screws_and_label(base_name='base', xray_path='xray.png',
                             selected_screws=selected_screws,
                             image_dir='./images', label_dir='./labels')
    visualize_segmentation('./images/base_drill.png',
                           './labels/base_drill.txt')
