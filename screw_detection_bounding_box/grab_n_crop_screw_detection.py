import cv2

from detect import run_model, add_random_point, x_crosshair, y_crosshair


def capture_video(device_index=0, target_width=None, target_height=None, crop_width=None, crop_height=None):
    # Open a connection to the USB camera

    cap = cv2.VideoCapture(device_index)

    # Check if the camera opened successfully

    if not cap.isOpened():
        print("Error: Could not open USB camera.")

        return

    try:

        # Set target resolution if provided

        if target_width and target_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)

            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

        # Get the adjusted resolution

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Adjusted Resolution: {width}x{height}")

        # Calculate the coordinates for the center crop

        crop_x = 256

        crop_y = 0

        # Create a full-screen window

        cv2.namedWindow("Screw Detection", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Screw Detection", cv2.WND_PROP_NORMAL, cv2.WINDOW_NORMAL)

        while True:

            # Capture frame-by-frame

            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture frame.")

                break

            # Crop the frame to the specified region

            side_crop = frame[0:crop_height, 0:256]
            cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

            # Flip the cropped frame horizontally to address mirroring

            # cropped_frame = cv2.flip(cropped_frame, 1)

            # Display the cropped frame in full screen
            img_cv = add_random_point(cropped_frame, x_crosshair, y_crosshair)
            updated_image = run_model(cropped_frame)
            im_h = cv2.hconcat([side_crop, updated_image])
            cv2.imshow("Screw Detection", im_h)

            # Exit on 'q' key press

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    finally:

        # Release the camera and close the OpenCV window

        cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Specify the USB camera index (you may need to change this based on your system)

    usb_camera_index = 0

    # Specify the target resolution (adjust as needed)

    target_width = 1280

    target_height = 1024

    # Specify the size of the crop region (adjust as needed)

    crop_width = 1024

    crop_height = 1024

    # Call the capture_video function with the specified resolution and crop region

    capture_video(usb_camera_index, target_width, target_height, crop_width, crop_height)