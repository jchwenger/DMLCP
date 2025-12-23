# --------------------------------------------------------------------------------

# Commands:
#   - 'q' to quit
#   - '1' to toggle hand colours
# --------------------------------------------------------------------------------

import urllib
import pathlib
import dataclasses

from typing import Tuple
from typing import Mapping

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model

from utils import DrawingSpec
from utils import draw_landmarks

# --------------------------------------------------------------------------------

WINDOW_NAME = "Gesture Recognition"
DESIRED_HEIGHT = 800
DESIRED_WIDTH = 600
DRAW_SUBSETS = False

MARGIN = 10  # pixels
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

# https://github.com/google-ai-edge/mediapipe/blob/9e4f898b22cf445c0ba7edc81ab4eb669fd71e89/mediapipe/python/solutions/hands_connections.py#L16
HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(
    *[
        HAND_PALM_CONNECTIONS,
        HAND_THUMB_CONNECTIONS,
        HAND_INDEX_FINGER_CONNECTIONS,
        HAND_MIDDLE_FINGER_CONNECTIONS,
        HAND_RING_FINGER_CONNECTIONS,
        HAND_PINKY_FINGER_CONNECTIONS,
    ]
)


# Function to draw landmarks and recognized gestures on the image
def draw_landmarks_and_gestures_on_image(
    bgr_image, recognition_result, DRAW_SUBSETS=False
):
    annotated_image = np.copy(bgr_image)

    if not getattr(recognition_result, "hand_landmarks", None):
        return annotated_image

    hand_landmarks_list = recognition_result.hand_landmarks
    handedness_list = recognition_result.handedness
    gesture_list = recognition_result.gestures

    # Loop through the detected hands and gestures to visualize
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        handedness = handedness_list[idx] if idx < len(handedness_list) else []
        corrected_handedness = (
            "Right" if handedness and handedness[0].category_name == "Left" else "Left"
        )
        gesture = gesture_list[idx] if idx < len(gesture_list) else []

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

        # Draw recognized gesture on the image
        if gesture:
            gesture_text = gesture[0].category_name
            cv2.putText(
                annotated_image,
                f"{gesture_text}",
                (text_x, text_y - 70),  # Display gesture above handedness
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                MAGENTA_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

    return annotated_image


# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/gesture_recognizer.task")
url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe GestureRecognizer
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

# --------------------------------------------------------------------------------

print("----------------------------------------")
print("Available gestures:")
print(
    "\n".join(
        [
            "None",
            "Closed_Fist ✊",
            "Open_Palm 👋",
            "Pointing_Up ☝️",
            "Thumb_Down 👎",
            "Thumb_Up 👍",
            "Victory  ✌️ ",
            "ILoveYou 🤟",
        ]
    )
)
print("----------------------------------------")


# Open webcam video stream
cap = cv2.VideoCapture(1)

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
    resized_frame = cv2.resize(frame, (DESIRED_HEIGHT, DESIRED_WIDTH))

    # Convert the frame to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform gesture recognition in the frame
    recognition_result = recognizer.recognize(mp_image)

    # Annotate frame with detected landmarks and gestures
    if recognition_result:
        annotated_frame = draw_landmarks_and_gestures_on_image(
            resized_frame, recognition_result, DRAW_SUBSETS
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
