# --------------------------------------------------------------------------------

# Audio classification with:
# https://ai.google.dev/edge/mediapipe/solutions/audio/audio_classifier

# --------------------------------------------------------------------------------

import time
import pathlib
import threading

import numpy as np

from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components.containers import audio_data
from mediapipe.tasks.python.audio.core import audio_task_running_mode
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model

from py5canvas import *
from py5canvas.canvas import hsv_to_rgb

# --------------------------------------------------------------------------------

VIDEO_SIZE = 512

# Audio recording parameters
SAMPLE_RATE = 16000  # Hz
NUM_CHANNELS = 1
BUFFER_SIZE = 15600  # ~1 second of audio at 16kHz
CLASSIFICATION_INTERVAL = 25  # only detect every n frames

# Waveform dimensions
WAVEFORM_HEIGHT = 420
RESULTS_Y_START = WAVEFORM_HEIGHT + 40

# YAMNet has 521 classes (from https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt)
NUM_CLASSES = 521

# Global state for results and timing
latest_result = None
inference_time_ms = 0.0
waveform_buffer = np.zeros(BUFFER_SIZE)
result_lock = threading.Lock()
audio_timestamp_ms = 0  # Timestamp for audio stream
last_request_time = {}  # Track request times by timestamp


def get_category_color(category_index, total_classes):
    """
    Get a consistent color for a category based on its index.
    Uses a 3D distribution in HSV color space to maximize color separation.
    Distributes colors across hue, saturation, and value dimensions using
    a method that ensures all three components vary independently.
    Returns RGB tuple (0-255 range) using py5canvas's hsv_to_rgb function.
    """
    # Use a 3D distribution that cycles through all dimensions
    # For 521 classes, we'll distribute across:
    # - Hue: primary dimension (most variation) - full 0-1 range
    # - Saturation: secondary dimension - 0.5-1.0 range (avoid too desaturated)
    # - Value: tertiary dimension - 0.6-1.0 range (avoid too dark)

    # Calculate how many steps we need for each dimension
    # We want roughly equal distribution, so we'll use the cube root
    # For 521: cube root ≈ 8.04, so we'll use 8 steps per dimension = 512 combinations
    steps = int(np.ceil(total_classes ** (1.0 / 3.0)))

    # Map index to 3D coordinates using modulo arithmetic
    # This ensures we cycle through all combinations
    h_step = category_index % steps
    s_step = (category_index // steps) % steps
    v_step = (category_index // (steps * steps)) % steps

    # Convert steps to normalized HSV values (0-1)
    # Hue: full range for maximum color variety
    hue_normalized = h_step / steps

    # Saturation: range 0.5-1.0 to keep colors vibrant
    saturation = 0.5 + (s_step / steps) * 0.5

    # Value: range 0.6-1.0 to keep colors bright enough
    value = 0.6 + (v_step / steps) * 0.4

    # Use py5canvas's hsv_to_rgb function directly to convert HSV to RGB
    # hsv_to_rgb expects normalized HSV values (0-1) and returns normalized RGB (0-1)
    rgb_normalized = hsv_to_rgb(np.array([hue_normalized, saturation, value]))
    # Convert to 0-255 range and return as tuple of integers
    return tuple(int(c * 255) for c in rgb_normalized[:3])


# Callback for classification results
def result_callback(result, timestamp_ms):
    """Callback function to receive classification results."""
    global latest_result, inference_time_ms
    with result_lock:
        latest_result = result
        # Calculate inference time from when we sent the request
        if timestamp_ms in last_request_time:
            inference_time_ms = (time.time() * 1000) - last_request_time[timestamp_ms]
            # Clean up old entries
            if len(last_request_time) > 10:
                oldest_key = min(last_request_time.keys())
                del last_request_time[oldest_key]


# --------------------------------------------------------------------------------

# Path to the model file
model_path = pathlib.Path("models/yamnet.tflite")
# see models here: https://ai.google.dev/edge/mediapipe/solutions/audio/audio_classifier#models
url = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe AudioClassifier with callback
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = audio.AudioClassifierOptions(
    base_options=base_options,
    max_results=-1,
    running_mode=audio_task_running_mode.AudioTaskRunningMode.AUDIO_STREAM,
    result_callback=result_callback,
)
classifier = audio.AudioClassifier.create_from_options(options)

# Create audio record for microphone input
audio_record = classifier.create_audio_record(NUM_CHANNELS, SAMPLE_RATE, BUFFER_SIZE)
audio_record.start_recording()

# --------------------------------------------------------------------------------


def setup():
    create_canvas(VIDEO_SIZE, VIDEO_SIZE)
    text_size(14)
    frame_rate(30)


def draw():
    global audio_timestamp_ms, waveform_buffer

    background(200, 220, 240)  # Light blue background

    # Read audio data from microphone
    try:
        audio_samples = audio_record.read(BUFFER_SIZE)
        # Convert to mono if stereo
        if len(audio_samples.shape) > 1:
            audio_samples = audio_samples[:, 0]

        # Update waveform buffer
        waveform_buffer = audio_samples.flatten()

        if frame_count % CLASSIFICATION_INTERVAL == 0:
            # Create AudioData and classify
            audio_data_obj = audio_data.AudioData.create_from_array(
                audio_samples, sample_rate=SAMPLE_RATE
            )

            # Classify the audio chunk (async - results come via callback)
            request_time = time.time() * 1000
            last_request_time[audio_timestamp_ms] = request_time
            classifier.classify_async(audio_data_obj, audio_timestamp_ms)
            # Note: inference_time_ms is updated in the callback

            # Update timestamp only when we actually classify
            # Increment by buffer duration in ms
            buffer_duration_ms = int((BUFFER_SIZE / SAMPLE_RATE) * 1000)
            audio_timestamp_ms += buffer_duration_ms

    except Exception as e:
        print(f"Error reading audio: {e}")

    # Draw inference time
    draw_inference_time()

    # Draw classification results
    draw_classification_results()

    # Draw waveform visualization
    draw_waveform()


def draw_waveform():
    """Draw the audio waveform visualization."""
    push()

    # Background for waveform area
    fill(200, 220, 240)
    no_stroke()
    rect((0, 0), (width, WAVEFORM_HEIGHT))

    # Center line
    stroke(64, 192, 192)  # Teal color
    stroke_weight(1)
    line((0, WAVEFORM_HEIGHT / 2), (width, WAVEFORM_HEIGHT / 2))

    # Draw waveform
    if len(waveform_buffer) > 0:
        stroke(64, 192, 192)  # Teal
        stroke_weight(1.5)
        no_fill()

        # Draw multiple overlapping lines for visual effect
        num_lines = 3
        for line_idx in range(num_lines):
            alpha = 255 // (line_idx + 1)
            stroke(64, 192, 192, alpha)

            # Draw waveform with slight offset for each line
            offset = (line_idx - 1) * 0.5
            begin_shape()
            no_fill()
            for i in range(width):
                buffer_idx = int((i / width) * len(waveform_buffer))
                if buffer_idx < len(waveform_buffer):
                    # Scale amplitude for visualization
                    amplitude = (
                        waveform_buffer[buffer_idx] * (WAVEFORM_HEIGHT / 2) * 0.8
                    )
                    y = (WAVEFORM_HEIGHT / 2) + amplitude + offset
                    vertex((i, y))
            end_shape()

    pop()


def draw_inference_time():
    """Draw the inference time text."""
    global inference_time_ms
    push()
    fill(0)
    no_stroke()
    text_size(14)
    text_align(RIGHT)
    with result_lock:
        current_time = inference_time_ms
    text(f"Inference time (ms): {current_time:.1f}", (width - 40, height - 10))
    pop()


def draw_classification_results():
    """Draw the classification results with bars."""
    push()

    with result_lock:
        if latest_result is None or not latest_result.classifications:
            pop()
            return

        # Get the first (and typically only) classification result
        classification = latest_result.classifications[0]
        categories = classification.categories

        # Sort by score (descending)
        sorted_categories = sorted(categories, key=lambda c: c.score, reverse=True)

        # Draw each category
        y_offset = RESULTS_Y_START
        bar_height = 30
        bar_spacing = 40
        bar_width = width - 80
        bar_x = 40

        for idx, category in enumerate(sorted_categories[:3]):
            # Category name and percentage
            category_name = category.category_name or f"Category {category.index}"
            score = category.score
            percentage = int(score * 100)

            # Truncate long names
            if len(category_name) > 30:
                category_name = category_name[:27] + "..."

            # Get consistent color for this category (based on index, not rank)
            # Uses HSV color space: hue = (index / NUM_CLASSES) * 360
            color = get_category_color(category.index, NUM_CLASSES)

            # Draw bar background (lighter version)
            fill(color[0] // 2, color[1] // 2, color[2] // 2, 180)
            no_stroke()
            rect((bar_x, y_offset), (bar_width, bar_height))

            # Draw filled portion
            fill(*color, 255)
            filled_width = bar_width * score
            rect((bar_x, y_offset), (filled_width, bar_height))

            # Draw text
            fill(255)
            text_size(12)
            text_align(LEFT)
            text(
                f"{category_name}: {percentage}%",
                (bar_x + 10, y_offset + bar_height / 2 + 4),
            )

            y_offset += bar_spacing

    pop()


def key_pressed(key):
    if key == " ":
        print("-" * 80)
        if len(latest_result.classifications) > 0:
            sorted_categories = sorted(
                latest_result.classifications[0].categories, key=lambda c: c.score
            )
            for c in sorted_categories:
                print(c)
        print("-" * 80)


run()
