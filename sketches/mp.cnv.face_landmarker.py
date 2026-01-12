# --------------------------------------------------------------------------------

# Face landmark detection using:
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

# --------------------------------------------------------------------------------

import pathlib

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model
from utils import landmarks_to_px

from utils import FACEMESH_LIPS
from utils import FACEMESH_NOSE

from utils import FACEMESH_IRISES
from utils import FACEMESH_LEFT_IRIS
from utils import FACEMESH_RIGHT_IRIS

from utils import FACEMESH_CONTOURS
from utils import FACEMESH_FACE_OVAL
from utils import FACEMESH_TESSELATION

from utils import FACEMESH_LEFT_EYE
from utils import FACEMESH_RIGHT_EYE

from utils import FACEMESH_LEFT_EYEBROW
from utils import FACEMESH_RIGHT_EYEBROW

from py5canvas import *

# --------------------------------------------------------------------------------


VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512

# Path to the model file
model_path = pathlib.Path("models/face_landmarker.task")
# see other models here: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe FaceLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
model = vision.FaceLandmarker.create_from_options(options)

# --------------------------------------------------------------------------------

video = VideoInput(size=(VIDEO_WIDTH, VIDEO_HEIGHT))


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)


def draw():
    background(0)

    # Video frame
    frame = video.read()
    # Convert to numpy 8 bit
    frame = np.array(frame)

    push()
    image(frame)

    # Convert the frame to RGB and create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # # Detect face landmarks in the frame
    result = model.detect(mp_image)

    if result and result.face_landmarks:
        # Convenience aliases to MediaPipe Face Mesh connection sets

        # Draw each detected face
        for lms in result.face_landmarks:
            pts = landmarks_to_px(lms, VIDEO_WIDTH, VIDEO_HEIGHT)

            no_fill()

            # 1) Light tessellation
            stroke(0, 255, 255, 100)  # cyan
            stroke_weight(0.4)
            draw_connections(pts, FACEMESH_TESSELATION)

            # 2) Accented contours (thicker) — different colors for readability
            stroke_weight(2)

            # Eyebrows
            stroke(255, 105, 180)  # pink
            draw_connections(pts, FACEMESH_LEFT_EYEBROW)
            stroke(186, 85, 211)  # purple
            draw_connections(pts, FACEMESH_RIGHT_EYEBROW)

            # Eyes
            stroke(65, 105, 225)  # blue
            draw_connections(pts, FACEMESH_LEFT_EYE)

            stroke(147, 112, 219)  # blue-purple
            draw_connections(pts, FACEMESH_RIGHT_EYE)

            stroke(255, 255, 0)  # yellow
            draw_connections(pts, FACEMESH_IRISES)

            # Lips
            stroke(255, 0, 0)  # red
            draw_connections(pts, FACEMESH_LIPS)

            # Face oval (white)
            stroke(255)
            draw_connections(pts, FACEMESH_FACE_OVAL)

            # # Face contours (eyes, eyebrows, mouth, around face)
            # stroke(0, 255, 0)
            # draw_connections(pts, FACEMESH_CONTOURS)

    pop()


# helpers ------------------------------------------------------------------------


# Helper: draw a set of connections
def draw_connections(pts, connections):
    for i, j in connections:
        line(pts[i], pts[j])


run()

# IDEAS, to make it your own:
# - There is no obligation to display the video, and you could for instance
#   imagine a blank canvas where a few points from the face are used to draw
#   vanishing circles, using the same logic as when you want a circle to leave
#   a trail behind it when it moves?
# - We have a `landmarks_to_px` function because the landmarks are
#   predicted in a *normalised* way (0-1). This means that instead of having
#   the mask drawn on the person's face, you could also work with a smaller,
#   fixed version of the face (e.g. in the upper left corner of the sketch), as
#   the values are restricted to be always in the same range. This in turn
#   could be used if you wanted a face mesh that moves like the person being
#   filmed, but that stays fixed (instead of being superimposed to the same
#   location in the image). Proprely scaled again, this 'static' yet moving
#   face could occupy the whole canvas, like a mirror!
# - Now we draw absolutely everything, and that could be a great starting point
#   to create some sort of mask, but of course it's possible to do something
#   different, using the various face points in different ways. It is probably
#   particularly interesting if you focus on only some points (maybe one in
#   each cheek? The corner of the eyes and/or the mouth? Or on the contrary
#   less obvious combinations?). Using three-four points would allow you to
#   define an arc, a spline, or a Bézier curve!
