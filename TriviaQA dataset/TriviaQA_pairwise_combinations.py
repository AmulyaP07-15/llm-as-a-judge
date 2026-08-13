import random
import itertools
import pandas as pd

RANDOM_SEED = 42
INPUT = "TriviaQA_model_answers.csv"
OUT = "TriviaQA_pairs.csv"

# column names in TriviaQA_model_answers.csv are prefixed with these
models = ["llama", "qwen", "allam"]

random.seed(RANDOM_SEED)

# input: TriviaQA_model_answers.csv (question + each model's answer)
# output: TriviaQA_pairs.csv, one row per model pair per question, order randomized
df = pd.read_csv(INPUT)

# build the full job list first so the swap flags can be split exactly in half
# rather than approximately. doing it per row with random() gives roughly 50
# percent but not exactly, which matters for a binomial test against 0.5.
jobs = []
for _, row in df.iterrows():
    for model_a, model_b in itertools.combinations(sorted(models), 2):
        answer_a = row[f"{model_a}_answer"]
        answer_b = row[f"{model_b}_answer"]

        # skip pairs where either model failed to answer
        if pd.isna(answer_a) or pd.isna(answer_b):
            continue

        jobs.append({
            "question": row["question"],
            "correct_answer": row["correct_answer"],
            "model_a": model_a,
            "answer_a": answer_a,
            "model_b": model_b,
            "answer_b": answer_b
        })

# exact half split, shuffled with a fixed seed so presentation order does not
# correlate with question order or with which models are being compared
total = len(jobs)
swap_flags = [True] * (total // 2) + [False] * (total - total // 2)
random.shuffle(swap_flags)

pairs = []
for job, swapped in zip(jobs, swap_flags):
    # model_a and model_b stay canonical and alphabetical so pair identity is
    # recoverable. first and second capture what the judge actually saw.
    if swapped:
        first_model, first_answer = job["model_b"], job["answer_b"]
        second_model, second_answer = job["model_a"], job["answer_a"]
    else:
        first_model, first_answer = job["model_a"], job["answer_a"]
        second_model, second_answer = job["model_b"], job["answer_b"]

    pairs.append({
        "question": job["question"],
        "correct_answer": job["correct_answer"],
        "model_a": job["model_a"],
        "model_b": job["model_b"],
        "swapped": swapped,
        "first_model": first_model,
        "first_answer": first_answer,
        "second_model": second_model,
        "second_answer": second_answer,
        # word counts as presented, used for the verbosity correlation
        "first_length_words": len(str(first_answer).split()),
        "second_length_words": len(str(second_answer).split())
    })

df_pairs = pd.DataFrame(pairs)
df_pairs.to_csv(OUT, index=False)

print(f"total pairs: {len(df_pairs)}")
print(f"swapped: {df_pairs['swapped'].sum()} of {len(df_pairs)}")
print(df_pairs[["model_a", "model_b", "swapped", "first_model"]].head())
print(f"saved to {OUT}")