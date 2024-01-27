import cv2

import ctypes

import numpy as np

from detect import run_model

WINDOW_NAME = 'Video Capture'

video_capture = cv2.VideoCapture(0)

if not video_capture.read()[0]:
    video_capture = cv2.VideoCapture(0)

# Full screen modea

cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)

cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while video_capture.isOpened():

    # get Screen Size

    user32 = ctypes.windll.user32

    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

    # read video frame by frame

    ret, frame = video_capture.read()
    frame.shape = (1024, 1280, _)
    # ret, frame1 = video_capture1.read()

    # ret, frame2 = video_capture2.read()

    # ret, frame3 = video_capture3.read()

    width = int(video_capture.get(3))

    height = int(video_capture.get(4))

    image = np.zeros(frame.shape, np.uint8)
    # frame.shape = 1024, 1280, _
    # frame_height, frame_width, _ = frame.shape
    # frame_height, frame_width = 1024, 1280
    # print(screen_width, ":", frame_width)
    #
    # scaleWidth = float(screen_width) / float(frame_width)
    #
    # scaleHeight = float(screen_height) / float(frame_height)
    # print(screen_height, ":", frame_height)

    # Specify the target resolution (adjust as needed)

    # target_width = 1280
    #
    # target_height = 720

    # Specify the size of the crop region (adjust as needed)

    crop_width = 920

    crop_height = 580

    # if scaleHeight > scaleWidth:
    #
    #     imgScale = scaleWidth
    # else:
    #
    #     imgScale = scaleHeight

    #    cv2.circle(image, point,radius,colour,lineWidth)     #circle properties as arguments

    image = frame

    print('large screen')
    updated_image = run_model(image)
    cv2.imshow(WINDOW_NAME, updated_image)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture

video_capture.release()

cv2.destroyAllWindows()
