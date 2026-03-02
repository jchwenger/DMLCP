from typing import List

import numpy as np
import torch
import gradio as gr
import imageio

# from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline

# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_ID = "CompVis/stable-diffusion-v1-4"

# Select the best available device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)

if device == "cuda":
    dtype = torch.bfloat16
elif device == "mps":
    dtype = torch.float16
else:
    dtype = torch.float32

print(f"Loading model '{MODEL_ID}' on device: {device} (dtype={dtype}).")

# pipe = StableDiffusionXLPipeline.from_pretrained(
pipe = StableDiffusionPipeline.from_pretrained(    
    MODEL_ID,
    torch_dtype=dtype,
)
pipe = pipe.to(device)


def _latents_to_pil(latents: torch.Tensor) -> List["PIL.Image.Image"]:
    """
    Decode latents to a list of PIL images via the VAE and image processor.
    """
    latents = latents / pipe.vae.config.scaling_factor
    with torch.inference_mode():
        decoded = pipe.vae.decode(latents).sample
    images = pipe.image_processor.postprocess(decoded, output_type="pil")
    return images


def _save_frames_to_video(frames, fps: int = 6) -> str:
    """
    Save a list of PIL images as a video file (mp4) and return its path.
    """
    if not frames:
        raise gr.Error("No frames generated, cannot create video.")

    video_path = "diffusion_gradual_denoising.mp4"

    # Convert PIL images to numpy arrays for imageio
    frames_np = [np.array(frame.convert("RGB")) for frame in frames]

    # imageio will use ffmpeg/imageio-ffmpeg under the hood if available
    imageio.mimsave(video_path, frames_np, fps=fps)

    return video_path


def generate_video(
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    fps: int = 6,
):
    """
    Generate a text-to-image video that visualizes the denoising process step by step.
    Returns the path to a video file for Gradio's Video component.
    """

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a non-empty text prompt.")

    try:
        num_inference_steps = int(num_inference_steps)
        guidance_scale = float(guidance_scale)
        fps = int(fps)
    except Exception as e:
        raise gr.Error(e)

    base_side = pipe.unet.config.sample_size * pipe.vae_scale_factor  # e.g. 64 * 8 = 512
    width = height = base_side
    print(f"Using base model resolution: {width = }, {height = }")

    with torch.inference_mode():
        frames: List["PIL.Image.Image"] = []

        def _callback(pipe, step: int, timestep: int, callback_kwargs):
            print(f"step {step + 1}/{num_inference_steps} (t={timestep})")
            latents = callback_kwargs["latents"]
            frame = _latents_to_pil(latents)[0]
            frames.append(frame)
            return callback_kwargs

        _ = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt
            if negative_prompt and negative_prompt.strip()
            else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            callback_on_step_end=_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        video_path = _save_frames_to_video(frames, fps=fps)

    return video_path, frames


demo = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="A watercolor painting of a robot reading a book",
            lines=2,
        ),
        gr.Textbox(
            label="Negative prompt",
            placeholder="Blur, low quality, distorted, artifacts",
            lines=2,
        ),
        gr.Slider(
            minimum=10,
            maximum=80,
            value=30,
            step=1,
            label="Number of denoising steps",
        ),
        gr.Slider(
            minimum=1.0,
            maximum=12.0,
            value=5.0,
            step=0.5,
            label="Guidance scale",
        ),
        gr.Slider(
            minimum=2,
            maximum=24,
            value=6,
            step=1,
            label="Video FPS",
        ),
    ],
    outputs=[
        gr.Video(label="Denoising video"),
        gr.Gallery(label="All frames", columns=6, height=400),
    ],
    title="Stable Diffusion – Gradual Denoising",
    description=(
        "This demo manually runs the Stable Diffusion denoising using a callback "
        "to capture each frames and returning a video."
    ),
)

demo.launch()