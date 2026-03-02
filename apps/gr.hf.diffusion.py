import torch
import gradio as gr

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
    use_safetensors=True,
)
pipe = pipe.to(device)


def generate_image(
    prompt: str,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1,
    width: int = 512,
    height: int = 512,
):
    """
    Text-to-image callback for Gradio with basic generation controls.
    """

    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a non-empty text prompt.")

    # Interpret width and height and validate.
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        raise gr.Error("Width and height must be integers.")

    # Basic safety/compatibility checks for Stable Diffusion.
    for side, name in ((w, "width"), (h, "height")):
        if side < 16 or side > 1024 or side % 8 != 0:
            raise gr.Error(
                f"{name.capitalize()} must be between 128 and 1024 pixels "
                "and divisible by 8 (e.g., 256, 384, 512, 640, 768, 896, 1024)."
            )

    width, height = w, h

    # Optional seeding for reproducibility; -1 means random each time
    generator = None
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    with torch.inference_mode():
        try:
            image = pipe(
                prompt=prompt,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                height=height,
                width=width,
                generator=generator,
            ).images[0]
        except Exception as e:
            # Catch and surface any size-related or pipeline errors cleanly to the UI.
            raise gr.Error(f"Generation failed (possibly due to invalid size): {e}")

    return image


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="A watercolor painting of a robot reading a book",
            lines=2,
        ),
        gr.Slider(
            minimum=1,
            maximum=100,
            value=30,
            step=1,
            label="Number of diffusion steps",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=20.0,
            value=7.5,
            step=0.1,
            label="Guidance scale (CFG)",
        ),
        gr.Number(
            value=-1,
            precision=0,
            label="Seed (-1 for random each run)",
        ),
        gr.Number(
            value=512,
            precision=0,
            label="Width (px, 16–1024, divisible by 8)",
        ),
        gr.Number(
            value=512,
            precision=0,
            label="Height (px, 16–1024, divisible by 8)",
        ),
    ],
    outputs=gr.Image(label="Generated image"),
    title="Stable Diffusion",
    description="Gradio app using Stable Diffusion via Hugging Face diffusers, with basic generation controls.",
)

demo.launch()
