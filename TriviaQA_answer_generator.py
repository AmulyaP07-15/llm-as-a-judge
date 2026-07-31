import os
import time
import pandas as pd
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# three models from distinct families so self enhancement bias is measurable
models = {
    "llama": "llama-3.1-8b-instant",
    "qwen": "qwen/qwen3.6-27b",
    "allam": "allam-2-7b"
}

INPUT = "TriviaQA_sampled.csv"
OUT = "TriviaQA_model_answers.csv"
BATCH_SIZE = 50

df = pd.read_csv(INPUT)

# resume from checkpoint if a previous run was interrupted
if os.path.exists(OUT):
    df_existing = pd.read_csv(OUT)
    done_questions = set(df_existing["question"].tolist())
    results = df_existing.to_dict("records")
    print(f"resuming from {len(results)} already done")
else:
    done_questions = set()
    results = []

total_tokens = 0

for i, row in df.iterrows():
    question = row["question"]

    if question in done_questions:
        continue

    entry = {"question": question, "correct_answer": row["answer_text"]}

    for model_name, model_id in models.items():
        print(f"querying {model_name} for question {i+1}...")

        kwargs = {
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": f"Answer in one short phrase only, no explanation: {question}"
            }],
            "max_tokens": 50,
            "temperature": 0
        }

        # qwen reasons before answering by default. left alone it returns a
        # truncated think block instead of an answer and burns roughly 2500
        # tokens a call, which exhausts the daily budget in a few hundred rows.
        if model_name == "qwen":
            kwargs["reasoning_effort"] = "none"

        try:
            response = client.chat.completions.create(**kwargs)
            entry[f"{model_name}_answer"] = response.choices[0].message.content.strip()
            if getattr(response, "usage", None):
                total_tokens += response.usage.total_tokens
        except Exception as e:
            print(f"  error with {model_name}: {e}")
            entry[f"{model_name}_answer"] = None

        # stay under the 30 requests per minute limit
        time.sleep(2)

    results.append(entry)

    if len(results) % BATCH_SIZE == 0:
        pd.DataFrame(results).to_csv(OUT, index=False)
        print(f"checkpoint saved at {len(results)} questions, {total_tokens:,} tokens")

pd.DataFrame(results).to_csv(OUT, index=False)
print(f"done. saved {len(results)} questions to {OUT}")
print(f"total tokens used: {total_tokens:,}")