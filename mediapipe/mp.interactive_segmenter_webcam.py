# --------------------------------------------------------------------------------
# Commands:
#   - '1' to toggle foreground mask
#   - '2' to toggle background mask
#     (0: no mask, 1: mask only, 2: mask with transparency)
# --------------------------------------------------------------------------------

import math
import urllib
import pathlib

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers

from utils import _normalized_to_pixel_coordinates

# --------------------------------------------------------------------------------

# Variables to control transparency
# 0: no mask, 1: mask only, 2: mask with transparency
foreground_display_mode = 0
background_display_mode = 0

# Default initial selection point
keypoint_x, keypoint_y = 0.5, 0.5

# Constants for segmentation
BG_COLOR = (255, 0, 255)  # magenta
FG_COLOR = (0, 255, 255)  # cyan

RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# Display configuration
WINDOW_NAME = "Interactive Segmentation"
DESIRED_HEIGHT = 800
DESIRED_WIDTH = 600


# Callback function to select point on mouse click
def click_event(event, x, y, flags, param):
    global keypoint_x, keypoint_y
    if event == cv2.EVENT_LBUTTONDOWN:
        keypoint_x, keypoint_y = x / param[0], y / param[1]


# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/interactive_segmenter.tflite")

# Check if the model file exists, if not, download it
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite"
    print()
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")

# Initialize ImageSegmenter
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.InteractiveSegmenterOptions(
    base_options=base_options, output_category_mask=True
)
segmenter = vision.InteractiveSegmenter.create_from_options(options)

# --------------------------------------------------------------------------------

# Open webcam video stream
cap = cv2.VideoCapture(1)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame horizontally (like a mirror)
    frame = cv2.flip(frame, 1)

    resized_frame = cv2.resize(frame, (DESIRED_HEIGHT, DESIRED_WIDTH))
    h, w = resized_frame.shape[:2]

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform segmentation on the selected keypoint
    roi = RegionOfInterest(
        format=RegionOfInterest.Format.KEYPOINT,
        keypoint=NormalizedKeypoint(keypoint_x, keypoint_y),
    )
    segmentation_result = segmenter.segment(mp_image, roi)
    category_mask = segmentation_result.category_mask

    # Convert mask to a boolean condition with same channel shape as the image
    mask = category_mask.numpy_view()
    if mask.ndim == 2:  # ensure channel dimension exists
        mask = np.expand_dims(mask, axis=-1)
    condition = np.repeat(mask, 3, axis=2) > 0.1

    # Prepare the foreground (cyan) and background (magenta) overlays
    fg_overlay = np.zeros(rgb_frame.shape, dtype=np.uint8)
    fg_overlay[:] = FG_COLOR
    bg_overlay = np.zeros(rgb_frame.shape, dtype=np.uint8)
    bg_overlay[:] = BG_COLOR

    # Apply the foreground mask based on the selected display mode
    if foreground_display_mode == 1:  # Only mask (fully opaque)
        output_image = np.where(condition, fg_overlay, rgb_frame)
    elif foreground_display_mode == 2:  # Mask with transparency
        output_image = cv2.addWeighted(fg_overlay, 0.5, rgb_frame, 0.5, 0)
        output_image = np.where(condition, output_image, rgb_frame)
    else:  # No mask, original frame
        output_image = rgb_frame

    # Apply the background mask based on the selected display mode
    if background_display_mode == 1:  # Only mask (fully opaque)
        output_image = np.where(~condition, bg_overlay, output_image)
    elif background_display_mode == 2:  # Mask with transparency
        temp_bg_image = cv2.addWeighted(bg_overlay, 0.5, rgb_frame, 0.5, 0)
        output_image = np.where(~condition, temp_bg_image, output_image)

    # Draw a white dot with black border to denote the point of interest
    keypoint_px = _normalized_to_pixel_coordinates(keypoint_x, keypoint_y, w, h)
    cv2.circle(output_image, keypoint_px, 6 + 5, (0, 0, 0), -1)
    cv2.circle(output_image, keypoint_px, 6, (255, 255, 255), -1)

    # Convert back to BGR for OpenCV display
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Display the output segmentation in real-time
    cv2.imshow(WINDOW_NAME, output_image)

    # Set mouse callback to capture click events
    cv2.setMouseCallback(WINDOW_NAME, click_event, param=(w, h))

    # Capture keypress to toggle foreground/background display modes
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):  # Exit on 'q'
        break
    elif key == ord("1"):  # Toggle foreground mask
        foreground_display_mode = (
            foreground_display_mode + 1
        ) % 3  # Cycles through 0, 1, 2
    elif key == ord("2"):  # Toggle background mask
        background_display_mode = (
            background_display_mode + 1
        ) % 3  # Cycles through 0, 1, 2

cap.release()
cv2.destroyAllWindows()
