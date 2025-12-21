# --------------------------------------------------------------------------------
# Commands:
#   - '1' to toggle pose segmentation mask
#     (0: no mask, 1: mask only, 2: mask with transparency)
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

# Segmentation mask display modes
# 0: No mask, 1: Transparent overlay, 2: Only mask
segmentation_mode = 0

# --------------------------------------------------------------------------------

# Drawing constants
WHITE_COLOR = (224, 224, 224)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
_BGR_CHANNELS = 3
_VISIBILITY_THRESHOLD = 0.5
_PRESENCE_THRESHOLD = 0.5

# https://github.com/google-ai-edge/mediapipe/blob/9e4f898b22cf445c0ba7edc81ab4eb669fd71e89/mediapipe/python/solutions/pose_connections.py#L16
POSE_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ]
)


# https://github.com/google-ai-edge/mediapipe/blob/9e4f898b22cf445c0ba7edc81ab4eb669fd71e89/mediapipe/python/solutions/drawing_utils.py#L39
@dataclasses.dataclass
class DrawingSpec:
    """Drawing style spec."""

    color: Tuple[int, int, int] = WHITE_COLOR
    thickness: int = 2
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
    normalized_x, normalized_y, image_width: int, image_height: int
):
    """Converts normalized value pair to pixel coordinates."""
    clamped_x = min(max(normalized_x, 0.0), 1.0)
    clamped_y = min(max(normalized_y, 0.0), 1.0)
    x_px = min(int(np.floor(clamped_x * image_width)), image_width - 1)
    y_px = min(int(np.floor(clamped_y * image_height)), image_height - 1)
    return (x_px, y_px)


# https://github.com/google-ai-edge/mediapipe/blob/9e4f898b22cf445c0ba7edc81ab4eb669fd71e89/mediapipe/python/solutions/drawing_utils.py#L119
def draw_landmarks(
    image,
    landmark_list,
    connections=None,
    landmark_drawing_spec=DrawingSpec(color=GREEN_COLOR),
    connection_drawing_spec=DrawingSpec(color=BLUE_COLOR, thickness=2, circle_radius=0),
    is_drawing_landmarks=True,
):
    """Draws the landmarks and the connections on the image.

    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmark_list: A normalized landmark list proto message to be annotated on
        the image.
      connections: A list of landmark index tuples that specifies how landmarks to
        be connected in the drawing.
      landmark_drawing_spec: Either a DrawingSpec object or a mapping from hand
        landmarks to the DrawingSpecs that specifies the landmarks' drawing
        settings such as color, line thickness, and circle radius. If this
        argument is explicitly set to None, no landmarks will be drawn.
      connection_drawing_spec: Either a DrawingSpec object or a mapping from hand
        connections to the DrawingSpecs that specifies the connections' drawing
        settings such as color and line thickness. If this argument is explicitly
        set to None, no landmark connections will be drawn.
      is_drawing_landmarks: Whether to draw landmarks. If set false, skip drawing
        landmarks, only contours will be drawed.

    Raises:
      ValueError: If one of the followings:
        a) If the input image is not three channel BGR.
        b) If any connetions contain invalid landmark index.
    """
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
        # Draws the connections if the start and end landmarks are both visible.
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

    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if is_drawing_landmarks and landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = (
                landmark_drawing_spec[idx]
                if isinstance(landmark_drawing_spec, Mapping)
                else landmark_drawing_spec
            )
            # White circle border
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
            # Fill color into the circle
            cv2.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )


# Function to draw landmarks on the image
def draw_landmarks_on_image(bgr_image, detection_result):
    annotated_image = np.copy(bgr_image)

    if not getattr(detection_result, "pose_landmarks", None):
        return annotated_image

    for pose_landmarks in detection_result.pose_landmarks:
        draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=POSE_CONNECTIONS,
            landmark_drawing_spec=DrawingSpec(
                color=GREEN_COLOR, thickness=2, circle_radius=2
            ),
            connection_drawing_spec=DrawingSpec(
                color=BLUE_COLOR, thickness=2, circle_radius=0
            ),
        )

    return annotated_image


# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/pose_landmarker.task")
model_path.parent.mkdir(exist_ok=True)

# Check if the model file exists, if not, download it
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")

# Initialize PoseLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

# --------------------------------------------------------------------------------

# Open webcam video stream
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame horizontally (like a mirror)
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB format (as expected by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect pose landmarks in the current frame
    detection_result = detector.detect(mp_image)

    # Draw pose landmarks on the frame (BGR image)
    annotated_frame = draw_landmarks_on_image(frame, detection_result)

    # Handle segmentation mask based on the current mode
    if detection_result.segmentation_masks:
        # Segmentation mask arrives in model output resolution; resize to frame.
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        frame_h, frame_w = annotated_frame.shape[:2]
        mask_8u = (segmentation_mask * 255).astype(np.uint8)
        if mask_8u.shape[:2] != (frame_h, frame_w):
            mask_8u = cv2.resize(
                mask_8u, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR
            )
        # Expand to 3 channels for blending or display.
        mask_rgb = np.repeat(mask_8u[:, :], 3, axis=2)

        if segmentation_mode == 1:  # Transparent overlay
            # Blend the segmentation mask with the frame (semi-transparent overlay)
            alpha = 0.5  # Transparency factor
            annotated_frame = cv2.addWeighted(
                annotated_frame, 1 - alpha, mask_rgb, alpha, 0
            )

        elif segmentation_mode == 2:  # Only the segmentation mask
            annotated_frame = mask_rgb

    # Display the annotated frame
    cv2.imshow("Pose Detection", annotated_frame)

    # Capture keypress to toggle segmentation mask display mode
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):  # Exit on 'q'
        break
    elif key == ord("1"):  # Toggle segmentation mask display mode
        segmentation_mode = (segmentation_mode + 1) % 3  # Cycles through 0, 1, 2

cap.release()
cv2.destroyAllWindows()
