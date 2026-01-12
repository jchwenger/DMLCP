# --------------------------------------------------------------------------------

# Hand landmarks and handedness label detection with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

# Commands:
#   - 'q' to quit
#   - '1' to toggle hand colours
# --------------------------------------------------------------------------------

import pathlib
import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model

from utils import DrawingSpec
from utils import draw_landmarks

from utils import HAND_PALM_CONNECTIONS
from utils import HAND_THUMB_CONNECTIONS
from utils import HAND_INDEX_FINGER_CONNECTIONS
from utils import HAND_MIDDLE_FINGER_CONNECTIONS
from utils import HAND_RING_FINGER_CONNECTIONS
from utils import HAND_PINKY_FINGER_CONNECTIONS
from utils import HAND_CONNECTIONS

# --------------------------------------------------------------------------------

WINDOW_NAME = "Hand Detection"

VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512

DRAW_SUBSETS = False

MARGIN = 10
FONT_SIZE = 2
FONT_THICKNESS = 2

# https://github.com/google-ai-edge/mediapipe/blob/9e4f898b22cf445c0ba7edc81ab4eb669fd71e89/mediapipe/python/solutions/drawing_utils.py#L32
WHITE_COLOR = (224, 224, 224)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
MAGENTA_COLOR = (255, 0, 255)
CYAN_COLOR = (255, 255, 0)
YELLOW_COLOR = (0, 255, 255)


# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result, DRAW_SUBSETS=False):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        corrected_handedness = (
            "Right" if handedness[0].category_name == "Left" else "Left"
        )

        if DRAW_SUBSETS:
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_PALM_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(
                    color=MAGENTA_COLOR, thickness=2, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_THUMB_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(
                    color=CYAN_COLOR, thickness=2, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_INDEX_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(
                    color=YELLOW_COLOR, thickness=2, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_MIDDLE_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(
                    color=BLUE_COLOR, thickness=2, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_RING_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(
                    color=GREEN_COLOR, thickness=2, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_PINKY_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(
                    color=RED_COLOR, thickness=2, circle_radius=0
                ),
                is_drawing_landmarks=False,
            )
            # Draw landmarks once after connections for clarity.
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=None,
                landmark_drawing_spec=DrawingSpec(
                    color=WHITE_COLOR, thickness=2, circle_radius=2
                ),
            )
        else:
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_CONNECTIONS,
                landmark_drawing_spec=DrawingSpec(
                    color=GREEN_COLOR, thickness=2, circle_radius=2
                ),
                connection_drawing_spec=DrawingSpec(
                    color=BLUE_COLOR, thickness=2, circle_radius=0
                ),
            )

        # Get the top left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image
        cv2.putText(
            annotated_image,
            f"{corrected_handedness}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            MAGENTA_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/hand_landmarker.task")
# see models here: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe HandLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --------------------------------------------------------------------------------

# Open webcam video stream
cap = cv2.VideoCapture(0)

# reshape window & no GUI
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Resize the frame to the desired dimensions
    resized_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

    # Convert the frame to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks in the frame
    detection_result = detector.detect(mp_image)

    # Annotate frame with detected landmarks
    if detection_result:
        annotated_frame = draw_landmarks_on_image(
            resized_frame, detection_result, DRAW_SUBSETS
        )
    else:
        annotated_frame = resized_frame

    # Display the annotated frame
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    if key == ord("1"):
        DRAW_SUBSETS = not DRAW_SUBSETS

cap.release()
cv2.destroyAllWindows()
