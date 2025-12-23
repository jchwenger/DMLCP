# Sketches

## Canvas

P5.js-like Python library (works in Jupyter Notebooks as well)

[Repo](https://github.com/colormotor/py5canvas), [examples](https://github.com/colormotor/py5canvas/tree/main/examples).

### Installation

With conda, this comes with the other dependencies of this repo (see also [here](https://github.com/colormotor/py5canvas?tab=readme-ov-file#installing-dependencies-with-conda), in case).

### Running sketches

```bash
(dmlcp) $ python basic_animation.py
```

See also the cool [gui](https://github.com/colormotor/py5canvas?tab=readme-ov-file#gui-support-and-parameters).

```bash
(dmlcp) $ python parameters.py
```


## Mediapipe

Examples of Python scripts using [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide), mostly adapted from [the example repo](https://github.com/google-ai-edge/mediapipe-samples) (and GPT...).

### Installation

With conda, this comes with the other dependencies of this repo (see also [here](https://github.com/colormotor/py5canvas?tab=readme-ov-file#installing-dependencies-with-conda), in case).

Then for instance:

```bash
(dmlcp) python pose_landmarker_webcam.py
```

The scripts are designed to download the model they need if they don't find it in the directory.

### Note: Ruff

I like to keep my Python code tidy, and using external programs can help you with that. I recommend [ruff](https://docs.astral.sh/ruff/) ([black](https://github.com/psf/black) is great, too), which is also included in this repo's environment.

With it, you can do: `ruff check` to check for syntax, and `ruff format` for formatting/cleaning your code.
