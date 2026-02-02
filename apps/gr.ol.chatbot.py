import ollama
import gradio as gr

# Name of the Ollama model to use. Change this to any model
# you have available in your local Ollama install:
# MODEL_ID = "gemma3:270m"
MODEL_ID = "qwen3:0.6b"

# Thinking models will output special thinking tokens, that we can display in
# various ways – with gemma3, flip this to False
THINK = True

# The overall 'directive' for our bot, see below
SYSTEM = "You are a helpful, concise assistant."

# test if the model is downloaded, if not pull (download) from the server
if MODEL_ID not in [m.model for m in ollama.list().models]:
    print(f"model '{MODEL_ID}' not found, downloading...")
    try:
        ollama.pull(MODEL_ID)
    except ollama.ResponseError as e:
        print(e)
        print(type(e))
        print("-" * 80)
        print("unable to pull model, aborting...")
        exit()


def predict(message, history):
    """
    Gradio ChatInterface callback.

    - `history` is a list of dicts with `role` and `content` (type="messages").
    - We append the latest user message.
    """

    # Build the conversation messages for Ollama's chat API.
    # We treat `history` + latest `message` as the conversation
    # and inject a single system prompt as the first message.

    # Start with the system message, then extend with previous turns,
    # then append the newest user message.

    # print(history)
    # breakpoint()

    messages = []

    if SYSTEM:
        messages.append({"role": "system", "content": SYSTEM})

    # `history` from Gradio ChatInterface(type="messages") is already
    # a list of {"role": ..., "content": ...} dicts.
    if history:
        for turn in history:
            # print(turn)
            role = turn.get("role", "user")
            content = turn.get("content", [])
            # the messages are actually subdivided into lists of various types
            # TODO: in a more advanced app, one could handle images or files!
            for subcontent in content:
                # print("subcontent:", subcontent)
                content_type = str(turn.get("type", ""))
                if content_type == "text":
                    content_text = str(turn.get("text", ""))
                    messages.append({"role": role, "text": content_text})

    # breakpoint()
    messages.append({"role": "user", "content": str(message)})

    # print()
    # print(messages)
    # print()
    # breakpoint()

    # Call Ollama with streaming enabled so we can yield partial tokens.
    stream = ollama.chat(
        model=MODEL_ID,
        messages=messages,
        stream=True,
        think=THINK,
    )

    generated = ""
    in_think = False

    for chunk in stream:
        # print(chunk)

        # Each chunk contains the incremental message content.
        msg = chunk.get("message", {})
        current_chunk = msg.get("content", "")
        current_thinking = msg.get("thinking", "")

        # no content and no thinking? continue
        if not (current_chunk or current_thinking):
            # print("no content nor thinking")
            continue

        # if it's not thinking, yet there's thinking in the chunk
        if current_thinking and not in_think:
            # print("activating thinking")
            in_think = True
            # wrap thinking in html
            generated += "<p style='color:#777; font-size: 12px; font-style:italic;'>"

        # if it's thinking, yet there's no longer any thinking in the chunk
        if not current_thinking and in_think:
            # print("deactivating thinking")
            in_think = False
            generated += "</p>"

        # Yield the growing assistant reply so Gradio can stream it.
        if in_think:
            generated += current_thinking
            # If within thinking tags, temporarily close the div for coherence
            yield generated + "</p>"
        else:
            generated += current_chunk
            # The thinking is over, the tag is closed
            yield generated


demo = gr.ChatInterface(
    predict,
    api_name="chat",
)

demo.launch()
