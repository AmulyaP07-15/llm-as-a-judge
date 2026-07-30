import time

from groq import Groq

client = Groq()

MODEL_IDS = {
    "llama": "llama-3.1-8b-instant",
    "qwen": "qwen/qwen3.6-27b",
    "allam": "allam-2-7b",
}


def call_groq(model_name, messages, max_tokens=150):
    model_id = MODEL_IDS[model_name]
    kwargs = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
    if model_name == "qwen":
        kwargs["reasoning_effort"] = "none"

    start = time.time()
    response = client.chat.completions.create(**kwargs)
    elapsed = time.time() - start

    return response.choices[0].message.content.strip(), elapsed