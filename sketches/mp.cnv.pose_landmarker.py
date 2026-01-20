# --------------------------------------------------------------------------------

# Pose landmarks with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

# Commands:
#   - '1' to toggle pose segmentation mask
#     (0: no mask, 1: mask only, 2: mask with transparency)
# --------------------------------------------------------------------------------

import pathlib
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model
from utils import landmarks_to_px

from utils import POSE_CONNECTIONS

from py5canvas import *

# --------------------------------------------------------------------------------


VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512

# Path to the model file
model_path = pathlib.Path("models/pose_landmarker_lite.task")
# see other models here: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models
url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe HandLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
model = vision.PoseLandmarker.create_from_options(options)

# --------------------------------------------------------------------------------

video = VideoInput(size=(VIDEO_WIDTH, VIDEO_HEIGHT))


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)


def draw():
    global result

    background(0)

    # Video frame
    frame = video.read()
    frame = np.array(frame)

    push()
    image(frame)

    # Detect pose landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = model.detect(mp_image)

    # Draw each detected person
    if result and result.pose_landmarks:
        for lms in result.pose_landmarks:
            pts = landmarks_to_px(lms, VIDEO_WIDTH, VIDEO_HEIGHT)

            # 1) Connections (white)
            no_fill()
            stroke(255)
            stroke_weight(2.0)
            draw_connections(pts, POSE_CONNECTIONS)

            # 2) Joints (cyan dots)
            no_stroke()
            fill(0, 200, 255)
            for x, y in pts:
                circle((x, y), 4.0)

    pop()  # close the push() done before drawing the frame


def mouse_pressed():
    global result

    print()
    # print(result)
    # print(result.__dict__.keys())
    # print()

    for pl in result.pose_landmarks:
        # print(pl)
        for ppl in pl:
            print(
                f"x: {ppl.x:.2f}, y: {ppl.y:.2f}, visibility: {ppl.visibility:.2f}, presence: {ppl.presence:.2f}, name: {ppl.name}"
            )


# helpers ------------------------------------------------------------------------


def draw_connections(pts, connections):
    for a, b in connections:
        line(pts[a], pts[b])


run()


# IDEAS, to make it your own:
# - There is no obligation to display the video, and you could for instance
#   imagine a blank canvas where a few points from the body are used to control
#   the position, or other parameters (the size?) of text that is displayed on
#   the screen, like in the Google and Bill T. Jones collaboration here:
#   https://experiments.withgoogle.com/billtjonesai
# - You could instead use the distance between, say, the nose and one wrist,
#   that would allow a user to control the size of a figure!, or to make
#   something happen if they touch their noses! In this project, this idea is
#   used to allow users to scream silently!
#   https://x.com/JordanneChan/status/1483988494032805895
# - Going further with this, you have no obligation to work with the entire
#   skeleton, and perhaps you might want to use only the two wrists, or the two
#   knees, or less obvious combinations (eye and hip?, wrist and knee?)
