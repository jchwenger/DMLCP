# LLM Inference

This works only on a WebGPU-enabled browser, like Chrome.

As per instructions [here](https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/README.md):

In order to begin, you must have a model available. You can download [Gemma 3n
E4B](https://huggingface.co/google/gemma-3n-E4B-it-litert-lm/blob/main/gemma-3n-E4B-it-int4-Web.litertlm), [Gemma 3n E2B](https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/blob/main/gemma-3n-E2B-it-int4-Web.litertlm), or [Gemma 3 1B](https://huggingface.co/litert-community/Gemma3-1B-IT), or
browse for more pre-converted models on our [LiteRT HuggingFace community](https://huggingface.co/litert-community/models), where files named "-web.task" are
specially converted to run optimally in the browser. All text-only variants of
Gemma 3 are available there, as well as [MedGemma-27B-Text](https://huggingface.co/litert-community/MedGemma-27B-IT/blob/main/medgemma-27b-it-int8-web.task). See
our web inference [guide](https://developers.google.com/mediapipe/solutions/genai/llm_inference/web_js) for more information.
