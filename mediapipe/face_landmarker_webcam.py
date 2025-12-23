# --------------------------------------------------------------------------------
# Commands:
#   - 'q' to quit
# --------------------------------------------------------------------------------

import urllib
import pathlib

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import resize_frame

from utils import DrawingSpec
from utils import draw_landmarks

from utils import get_default_face_mesh_contours_style
from utils import get_default_face_mesh_tesselation_style
from utils import get_default_face_mesh_iris_connections_style

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

# --------------------------------------------------------------------------------

WINDOW_NAME = "Face Detection"
DESIRED_HEIGHT = 1280
DESIRED_WIDTH = 832

# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if not getattr(detection_result, "face_landmarks", None):
        return annotated_image

    for face_landmarks in detection_result.face_landmarks:
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=get_default_face_mesh_tesselation_style(),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=get_default_face_mesh_contours_style(),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=get_default_face_mesh_iris_connections_style(),
        )
        # Additional feature-specific overlays
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_LIPS,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(
                color=(0, 0, 255), thickness=2, circle_radius=0
            ),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_NOSE,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(
                color=(0, 255, 255), thickness=1, circle_radius=0
            ),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_FACE_OVAL,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(
                color=(0, 255, 0), thickness=2, circle_radius=0
            ),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(
                color=(255, 0, 0), thickness=1, circle_radius=0
            ),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(
                color=(255, 0, 0), thickness=1, circle_radius=0
            ),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_LEFT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(
                color=(255, 255, 0), thickness=2, circle_radius=0
            ),
        )
        draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FACEMESH_RIGHT_EYEBROW,
            landmark_drawing_spec=None,
            connection_drawing_spec=DrawingSpec(
                color=(255, 255, 0), thickness=2, circle_radius=0
            ),
        )

    return annotated_image


# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/face_landmarker.task")
model_path.parent.mkdir(exist_ok=True)

# Check if the model file exists, if not, download it
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    print()
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")

# Initialize MediaPipe FaceLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(options)

# --------------------------------------------------------------------------------

# Open webcam video stream
cap = cv2.VideoCapture(1)

# reshape window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a natural webcam experience
    frame = cv2.flip(frame, 1)

    # Resize the frame to the desired dimensions
    resized_frame = resize_frame(frame, DESIRED_WIDTH, DESIRED_HEIGHT)

    # Convert the frame to RGB and create MediaPipe Image
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect face landmarks in the frame
    detection_result = detector.detect(mp_image)

    # Annotate frame with detected landmarks
    if detection_result:
        annotated_frame = draw_landmarks_on_image(resized_frame, detection_result)

        # Show the facial transformation matrix for debugging
        # print(detection_result.facial_transformation_matrixes)
    else:
        annotated_frame = resized_frame

    # Display the annotated frame
    cv2.imshow(WINDOW_NAME, annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
