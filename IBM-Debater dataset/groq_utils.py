import random
import time

from groq import Groq

client = Groq()

MODEL_IDS = {
    "llama": "llama-3.1-8b-instant",
    "qwen": "qwen/qwen3.6-27b",
    "allam": "allam-2-7b",
}

REQUEST_DELAY_SEC = 0.5
MAX_RETRIES = 5
BASE_BACKOFF_SEC = 2


def _is_rate_limit_error(e):
    """
    checks whether an exception looks like a groq rate limit error.

    input: an exception
    output: True if it looks like a 429
    """
    status_code = getattr(e, "status_code", None)
    return status_code == 429 or "429" in str(e) or "rate_limit" in str(e).lower()


def call_groq(model_name, messages, max_tokens=150):
    """
    sends a chat request to groq, retrying with backoff on rate limits.

    input: model key from MODEL_IDS, chat messages, max output tokens
    output: (reply text, seconds elapsed)
    """
    model_id = MODEL_IDS[model_name]
    kwargs = {"model": model_id, "messages": messages, "max_tokens": max_tokens}
    if model_name == "qwen":
        kwargs["reasoning_effort"] = "none"

    time.sleep(REQUEST_DELAY_SEC)

    for attempt in range(MAX_RETRIES):
        start = time.time()
        try:
            response = client.chat.completions.create(**kwargs)
            elapsed = time.time() - start
            return response.choices[0].message.content.strip(), elapsed
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF_SEC * (2 ** attempt) + random.uniform(0, 1)
                print(f"    rate limited, retrying in {wait:.1f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            raise