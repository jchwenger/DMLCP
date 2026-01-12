# --------------------------------------------------------------------------------

# Interactive segmentation with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/interactive_segmenter

# Commands:
# - click to change selection point
# - '1' to toggle foreground mask
# - '2' to toggle background mask
#   (0: no mask, 1: mask only, 2: mask with transparency)
# --------------------------------------------------------------------------------

import pathlib

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers

from utils import ensure_model

from py5canvas import *

# --------------------------------------------------------------------------------


VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600

# Variables to control transparency
# 0: no mask, 1: mask only, 2: mask with transparency
FOREGROUND_DISPLAY_MODE = 0
BACKGROUND_DISPLAY_MODE = 0

# Default initial selection point
KEYPOINT_X, KEYPOINT_Y = 0.5, 0.5

# Constants for segmentation
BG_COLOR = (255, 0, 255)  # magenta
FG_COLOR = (0, 255, 255)  # cyan

RegionOfInterest = vision.InteractiveSegmenterRegionOfInterest
NormalizedKeypoint = containers.keypoint.NormalizedKeypoint

# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/magic_touch.tflite")
url = "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/latest/magic_touch.tflite"
model_path = ensure_model(model_path, url)

# Initialize ImageSegmenter
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.InteractiveSegmenterOptions(
    base_options=base_options, output_category_mask=True
)
segmenter = vision.InteractiveSegmenter.create_from_options(options)

# --------------------------------------------------------------------------------

video = VideoInput(size=(VIDEO_WIDTH, VIDEO_HEIGHT))


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)
    text_size(12)


def draw():
    background(0)

    # Video frame
    frame = np.array(video.read())

    # Convert frame to MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Perform segmentation on the selected keypoint
    roi = RegionOfInterest(
        format=RegionOfInterest.Format.KEYPOINT,
        keypoint=NormalizedKeypoint(KEYPOINT_X, KEYPOINT_Y),
    )
    segmentation_result = segmenter.segment(mp_image, roi)
    category_mask = segmentation_result.category_mask

    # Convert mask to a boolean condition with same channel shape as the image
    mask = category_mask.numpy_view()
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    condition = np.repeat(mask, 3, axis=2) > 0.1

    # Prepare the foreground (cyan) and background (magenta) overlays
    fg_overlay = np.zeros(frame.shape, dtype=np.uint8)
    fg_overlay[:] = FG_COLOR
    bg_overlay = np.zeros(frame.shape, dtype=np.uint8)
    bg_overlay[:] = BG_COLOR

    output_image = frame.copy()

    # Apply the foreground mask based on the selected display mode
    if FOREGROUND_DISPLAY_MODE == 1:  # Only mask (fully opaque)
        output_image = np.where(condition, fg_overlay, output_image)
    elif FOREGROUND_DISPLAY_MODE == 2:  # Mask with transparency
        blended_fg = (0.5 * fg_overlay + 0.5 * frame).astype(np.uint8)
        output_image = np.where(condition, blended_fg, output_image)

    # Apply the background mask based on the selected display mode
    if BACKGROUND_DISPLAY_MODE == 1:  # Only mask (fully opaque)
        output_image = np.where(~condition, bg_overlay, output_image)
    elif BACKGROUND_DISPLAY_MODE == 2:  # Mask with transparency
        blended_bg = (0.5 * bg_overlay + 0.5 * frame).astype(np.uint8)
        output_image = np.where(~condition, blended_bg, output_image)

    # Draw to the canvas
    push()
    image(output_image)

    # Selection point indicator
    keypoint_px = (
        int(KEYPOINT_X * VIDEO_WIDTH),
        int(KEYPOINT_Y * VIDEO_HEIGHT),
    )
    stroke(0)
    stroke_weight(3)
    fill(255, 255, 255)
    circle(keypoint_px, 8)
    no_fill()
    stroke(0, 200, 255)
    stroke_weight(2)
    circle(keypoint_px, 12)

    pop()

    # draw_hud()


def key_pressed(key, mods=None):
    """Toggle masks with keys 1 (foreground) and 2 (background)."""
    global FOREGROUND_DISPLAY_MODE, BACKGROUND_DISPLAY_MODE
    if key == "1":
        FOREGROUND_DISPLAY_MODE = (FOREGROUND_DISPLAY_MODE + 1) % 3
    elif key == "2":
        BACKGROUND_DISPLAY_MODE = (BACKGROUND_DISPLAY_MODE + 1) % 3


def mouse_pressed(button, mods=None):
    """Update the segmentation keypoint to the mouse position."""
    global KEYPOINT_X, KEYPOINT_Y
    KEYPOINT_X = min(max(mouse_x / width, 0.0), 1.0)
    KEYPOINT_Y = min(max(mouse_y / height, 0.0), 1.0)


# helpers ------------------------------------------------------------------------


def draw_hud():
    """Small overlay explaining controls."""
    push()
    text_size(12)
    hud_lines = [
        "1: toggle foreground mask",
        "2: toggle background mask",
        "Click: move keypoint",
    ]
    x, y = 12, height - 12
    for line in reversed(hud_lines):
        draw_label((x, y), line)
        y -= 18
    pop()


def draw_label(anchor_xy, text_str):
    x, y = anchor_xy

    txt_pad_x, txt_pad_y = 6, 4
    text_size(12)
    tw = text_width(text_str)
    th = text_height(text_str)

    rect_mode(CORNER)
    no_stroke()
    fill(0, 0, 0, 190)
    rectangle(
        (x - txt_pad_x, y - th - txt_pad_y), (tw + 2 * txt_pad_x, th + 2 * txt_pad_y)
    )

    fill(255)
    text(text_str, (x, y))


run()
