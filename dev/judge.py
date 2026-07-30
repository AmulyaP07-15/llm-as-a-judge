import pandas as pd
import itertools
import random

random.seed(42)

df = pd.read_csv("model_answers.csv")

models = ["llama3.1", "mistral", "gemma"]
pairs = []

for _, row in df.iterrows():
    question = row["question"]
    correct = row["correct_answer"]

    # get all 3 pairwise combos for this question
    for model_a, model_b in itertools.combinations(models, 2):
        answer_a = row[f"{model_a}_answer"]
        answer_b = row[f"{model_b}_answer"]

        # randomly swap order in half the pairs for position bias testing
        swapped = random.random() > 0.5
        if swapped:
            model_a, model_b = model_b, model_a
            answer_a, answer_b = answer_b, answer_a

        pairs.append({
            "question": question,
            "correct_answer": correct,
            "model_a": model_a,
            "answer_a": answer_a,
            "len_a": len(str(answer_a).split()),  # word count for verbosity bias
            "model_b": model_b,
            "answer_b": answer_b,
            "len_b": len(str(answer_b).split()),
            "swapped": swapped
        })

df_pairs = pd.DataFrame(pairs)
print(f"total pairs: {len(df_pairs)}")
print(df_pairs.head())
df_pairs.to_csv("pairs.csv", index=False)
print("saved to pairs.csv")