# --------------------------------------------------------------------------------

# Object detection with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector

# Commands:
#   - 'q' to quit
# --------------------------------------------------------------------------------

import time
import pathlib

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import show_fps
from utils import ensure_model

# --------------------------------------------------------------------------------

WINDOW_NAME = "Object Detection"
VIDEO_SIZE = 512

SHOW_FPS = False
FPS_AVG_FRAME_COUNT = 10

MARGIN = 30
ROW_SIZE = 30
FONT_SIZE = 3
FONT_THICKNESS = 2
RECT_COLOR = (255, 0, 0)  # blue (BGR)
TEXT_COLOR = (255, 255, 255)  # white

model_path = pathlib.Path("models/efficientdet_lite0.tflite")
# see other models here: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector
url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
model_path = ensure_model(model_path, url)


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualized.
    Returns:
        Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, RECT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + " (" + str(probability) + ")"
        # print(result_text)
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image


# --------------------------------------------------------------------------------


detection_result_list = []


def visualize_callback(
    result: vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int
):
    result.timestamp_ms = timestamp_ms
    detection_result_list.append(result)


# Initialize ObjectDetector
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    score_threshold=0.5,
    result_callback=visualize_callback,
)
detector = vision.ObjectDetector.create_from_options(options)

# Variables to calculate FPS
counter, fps = 0, 0
start_time = time.time()

# Open webcam video stream
cap = cv2.VideoCapture(0)

# reshape window & no GUI
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    # For frame rate calculation
    counter += 1

    # Flip frame horizontally (like a mirror)
    frame = cv2.flip(frame, 1)

    # Resize the frame to the desired dimensions
    resized_frame = cv2.resize(frame, (VIDEO_SIZE, VIDEO_SIZE))

    # Convert frame to RGB format (as expected by MediaPipe)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run object detection using the model.
    detector.detect_async(mp_image, counter)
    current_frame = mp_image.numpy_view()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

    if SHOW_FPS:
        # Calculate the FPS
        if counter % FPS_AVG_FRAME_COUNT == 0:
            end_time = time.time()
            fps = FPS_AVG_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()

        show_fps(current_frame, fps)

    # Print result list on pressing 'p'
    if cv2.waitKey(5) & 0xFF == ord("p"):
        print(detection_result_list)

    if detection_result_list:
        # print(detection_result_list)
        annotated_frame = visualize(current_frame, detection_result_list[0])
        cv2.imshow(WINDOW_NAME, annotated_frame)
        detection_result_list.clear()
    else:
        cv2.imshow(WINDOW_NAME, current_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
