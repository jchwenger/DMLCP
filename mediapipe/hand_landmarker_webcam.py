import urllib
import pathlib

import dataclasses
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

# Drawing constants
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
MAGENTA_COLOR = (255, 0, 255)
CYAN_COLOR = (255, 255, 0)
YELLOW_COLOR = (0, 255, 255)
_BGR_CHANNELS = 3
_VISIBILITY_THRESHOLD = 0.5
_PRESENCE_THRESHOLD = 0.5

# Text overlay constants
MARGIN = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

# https://github.com/google-ai-edge/mediapipe/blob/9e4f898b22cf445c0ba7edc81ab4eb669fd71e89/mediapipe/python/solutions/hands_connections.py#L16
HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))

HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))

HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))

HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))

HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))

HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))

HAND_CONNECTIONS = frozenset().union(*[
    HAND_PALM_CONNECTIONS, HAND_THUMB_CONNECTIONS,
    HAND_INDEX_FINGER_CONNECTIONS, HAND_MIDDLE_FINGER_CONNECTIONS,
    HAND_RING_FINGER_CONNECTIONS, HAND_PINKY_FINGER_CONNECTIONS
])


@dataclasses.dataclass
class DrawingSpec:
    """Drawing style spec."""

    color: Tuple[int, int, int] = WHITE_COLOR
    thickness: int = 2
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Optional[Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    clamped_x = min(max(normalized_x, 0.0), 1.0)
    clamped_y = min(max(normalized_y, 0.0), 1.0)
    x_px = min(int(np.floor(clamped_x * image_width)), image_width - 1)
    y_px = min(int(np.floor(clamped_y * image_height)), image_height - 1)
    return (x_px, y_px)


def draw_landmarks(
    image: np.ndarray,
    landmark_list: Sequence,
    connections: Optional[Iterable[Tuple[int, int]]] = None,
    landmark_drawing_spec: Optional[
        Union[DrawingSpec, Mapping[int, DrawingSpec]]
    ] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
    ] = DrawingSpec(),
    is_drawing_landmarks: bool = True,
):
    """Draws landmarks and connections (adapted from mediapipe drawing_utils)."""
    if not landmark_list:
        return
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list):
        visibility = getattr(landmark, "visibility", 1.0) or 1.0
        presence = getattr(landmark, "presence", 1.0) or 1.0
        if visibility < _VISIBILITY_THRESHOLD or presence < _PRESENCE_THRESHOLD:
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark.x, landmark.y, image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        num_landmarks = len(landmark_list)
        for connection in connections:
            start_idx, end_idx = connection
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection from landmark "
                    f"#{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = (
                    connection_drawing_spec[connection]
                    if isinstance(connection_drawing_spec, Mapping)
                    else connection_drawing_spec
                )
                cv2.line(
                    image,
                    idx_to_coordinates[start_idx],
                    idx_to_coordinates[end_idx],
                    drawing_spec.color,
                    drawing_spec.thickness,
                )

    if is_drawing_landmarks and landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = (
                landmark_drawing_spec[idx]
                if isinstance(landmark_drawing_spec, Mapping)
                else landmark_drawing_spec
            )
            circle_border_radius = max(
                drawing_spec.circle_radius + 1,
                int(drawing_spec.circle_radius * 1.2),
            )
            cv2.circle(
                image,
                landmark_px,
                circle_border_radius,
                WHITE_COLOR,
                drawing_spec.thickness,
            )
            cv2.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )

# Path to the model file
model_path = pathlib.Path("models/hand_landmarker.task")

# Check if the model file exists, if not, download it
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    print()
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")

# Initialize MediaPipe HandLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result, draw_subsets=False):
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

        if draw_subsets:
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_PALM_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(color=MAGENTA_COLOR, thickness=2, circle_radius=0),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_THUMB_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(color=CYAN_COLOR, thickness=2, circle_radius=0),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_INDEX_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(color=YELLOW_COLOR, thickness=2, circle_radius=0),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_MIDDLE_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(color=BLUE_COLOR, thickness=2, circle_radius=0),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_RING_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=0),
                is_drawing_landmarks=False,
            )
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_PINKY_FINGER_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=DrawingSpec(color=RED_COLOR, thickness=2, circle_radius=0),
                is_drawing_landmarks=False,
            )
            # Draw landmarks once after connections for clarity.
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=None,
                landmark_drawing_spec=DrawingSpec(color=WHITE_COLOR, thickness=2, circle_radius=2),
            )
        else:
            draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                connections=HAND_CONNECTIONS,
                landmark_drawing_spec=DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=2),
                connection_drawing_spec=DrawingSpec(color=BLUE_COLOR, thickness=2, circle_radius=0),
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
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


# Open webcam video stream
cap = cv2.VideoCapture(1)
draw_subsets = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks in the frame
    detection_result = detector.detect(mp_image)

    # Annotate frame with detected landmarks
    if detection_result:
        annotated_frame = draw_landmarks_on_image(frame, detection_result, draw_subsets)
    else:
        annotated_frame = frame

    # Display the annotated frame
    cv2.imshow("Hand Detection", annotated_frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    if key == ord("1"):
        draw_subsets = not draw_subsets

cap.release()
cv2.destroyAllWindows()
