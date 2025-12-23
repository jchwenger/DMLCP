# --------------------------------------------------------------------------------
# Commands:
#   - 'q' to quit
# --------------------------------------------------------------------------------

import math
import urllib
import pathlib

import numpy as np

import cv2

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import resize_frame

WINDOW_NAME = "Image Classification"

# Height and width that will be used by the model
DESIRED_HEIGHT = 1280
DESIRED_WIDTH = 832

# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/classifier.tflite")
model_path.parent.mkdir(exist_ok=True)

# Check if the model file exists, if not, download it
if not model_path.exists():
    url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite"
    print()
    print(f"Downloading model from {url}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded and saved as {model_path}")

# Initialize MediaPipe ImageClassifier
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

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

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Resize the frame to the desired dimensions
    resized_frame = resize_frame(frame, DESIRED_WIDTH, DESIRED_HEIGHT)

    # Convert the frame to RGB format (as expected by MediaPipe)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image from the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Classify the frame
    classification_result = classifier.classify(mp_image)

    # Extract classification result (if available)
    if classification_result and classification_result.classifications:
        top_category = classification_result.classifications[0].categories[0]
        prediction_text = f"{top_category.category_name} ({top_category.score:.2f})"

        # Put the classification result on the frame
        cv2.putText(
            resized_frame,
            prediction_text,
            org=(30, 30),  # bottom left corner of text
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,  # antialiased line
        )

    # Display the frame with the prediction
    cv2.imshow(WINDOW_NAME, resized_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
