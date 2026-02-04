import ollama
import gradio as gr

# TODO: Render markdown properly.
#       Add an option to make the thinking visible or not? (There might be
#       more creative ways to handle that...)

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

    # print("history:")
    # print(history)
    # print()

    messages = []

    if SYSTEM:
        messages.append({"role": "system", "content": SYSTEM})

    # Gradio ChatInterface(type="messages") history: list of {role, content}.
    # content can be a string (text-only) or a list of parts (e.g. {type, text}).
    # We normalize to Ollama format: {role: "user"|"assistant"|"system", content: str}.
    if history:
        for turn in history:
            # print(f"{turn}")
            role = turn.get("role", "user")
            content = turn.get("content", [])
            # Various kinds of content (text, image, file path...)
            # TODO: develop this into a full multimodal chatbot, allowing users
            # to upload images, files, etc.
            for subcontent in content:
                # print(f" - {subcontent}")
                # Get the message type
                subcontent_type = subcontent.get("type", "")
                # If it is text, add to our messages
                if subcontent_type == "text":
                    content_text = subcontent.get("text", "")
                    messages.append({"role": role, "content": content_text})

    messages.append({"role": "user", "content": message})

    # print("messages:")
    # print(messages)
    # print()

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
