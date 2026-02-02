import threading

import torch
import gradio as gr

from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM
from transformers import TextIteratorStreamer
# from transformers import BitsAndBytesConfig

# BEWARE: this app will only work with 'chat' models (that have a
#         `.chat_template` in their `tokenizer` – you can check that
#         Qwen3-06B has one: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json)
#         Also, note that there is a mechanism to detect 'thinking' tokens and
#         displaying them differently, but if the chosen model outputs them in
#         a different format than <think></think>, then that won't work, and
#         you need to study the model output and change the checks accordingly!
# MODEL_ID = "google/gemma-3-270m-it"
MODEL_ID = "Qwen/Qwen3-0.6B"

device = (
    "cuda"
    if torch.cuda.is_available()
    # note: models using bfloat16 aren't compatible with MPS
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)

# Theoretically, you can reduce the memory footprint and increase the speed of
# your model by loading it quantized, but that means making sure bitsandbytes
# is installed (with pip only), and my tests haven't led to conclusive results
# quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    # quantization_config=quantization_config
).to(device)

# Context window from model config (fallback if missing)
context_window = getattr(model.config, "max_position_embeddings", None)
if context_window is None:
    context_window = getattr(tokenizer, "model_max_length", 2048)

print(f"model: {MODEL_ID}, context window: {context_window}.")


def predict(message, history):
    """
    Gradio ChatInterface callback.

    - `history` is a list of dicts with `role` and `content` (type="messages").
    - We append the latest user message, then build a chat template for Qwen.
    """

    print(history)

    # Make sure we don't mutate Gradio's history list in-place
    conversation = history + [{"role": "user", "content": message}]

    # Optionally prepend a system prompt; this also helps some Qwen templates.
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful, concise assistant.",
        },
        *conversation,
    ]

    # Use Qwen's chat template and add a generation prompt so the model knows
    # it should now produce the assistant's reply.
    input_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    # Set max_new_tokens to fill remaining context
    input_len = inputs["input_ids"].shape[1]
    max_new_tokens = max(1, context_window - input_len)

    # Set up a text streamer so we can yield partial generations
    # token-by-token (or small chunks), while the model runs in a
    # background thread.
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    generation_config.max_new_tokens = max_new_tokens
    # suppressing a pesky warning (https://stackoverflow.com/a/71397707)
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    # Run generation in a separate thread so that we can iterate over
    # the streamer in this function and yield updates to Gradio.
    def _run_generation():
        model.generate(
            **inputs,
            generation_config=generation_config,
            streamer=streamer,
        )

    thread = threading.Thread(target=_run_generation)
    thread.start()

    # Streamed parsing of the `<think>...</think>` block.
    # As soon as we see `<think>` in the stream, we start treating
    # everything that follows as "reasoning" until we encounter `</think>`.
    full_answer = ""
    in_think = False

    for new_text in streamer:
        if not new_text:
            continue

        # Wrap thinking in a p with dedicated html
        next_text_stripped = new_text.strip()
        if next_text_stripped == "<think>":
            full_answer += "<p style='color:#777; font-size: 12px; font-style:italic;'>"
            in_think = True
            continue
        if next_text_stripped == "</think>":
            full_answer += "</p>"
            in_think = False
            continue

        full_answer += new_text

        if in_think:
            # If within thinking tags, temporarily close the div for coherence
            yield full_answer + "</p>"
        else:
            # The thinking is over, the tag is closed
            yield full_answer

    # Ensure the generation thread is finished before returning.
    thread.join()


demo = gr.ChatInterface(
    predict,
    api_name="chat",
)

demo.launch()
