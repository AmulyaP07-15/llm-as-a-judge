import ollama
import pandas as pd
import time
import os

# load the full dataset
df = pd.read_csv("trivia_qa_sampled.csv")

models = ["llama3.1", "mistral", "gemma"]

# resume from existing results if file already exists
if os.path.exists("model_answers.csv"):
    df_existing = pd.read_csv("model_answers.csv")
    done_questions = set(df_existing["question"].tolist())
    results = df_existing.to_dict("records")
    print(f"resuming from {len(results)} already done")
else:
    done_questions = set()
    results = []

BATCH_SIZE = 50

for i, row in df.iterrows():
    question = row["question"]

    # skip if already done
    if question in done_questions:
        continue

    correct = row["answer_text"]
    entry = {"question": question, "correct_answer": correct}

    for model in models:
        print(f"querying {model} for question {i+1}...")
        try:
            response = ollama.chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": f"Answer in one short phrase only, no explanation: {question}"
                }]
            )
            entry[f"{model}_answer"] = response["message"]["content"].strip()
            print(f"{model} done")
        except Exception as e:
            print(f"error on row {i} with {model}: {e}")
            entry[f"{model}_answer"] = None

        time.sleep(0.5)

    results.append(entry)

    # save after every batch
    if len(results) % BATCH_SIZE == 0:
        pd.DataFrame(results).to_csv("model_answers.csv", index=False)
        print(f"checkpoint saved at {len(results)} questions")

# final save
pd.DataFrame(results).to_csv("model_answers.csv", index=False)
print(f"done. saved {len(results)} questions to model_answers.csv")