import csv
import random
import itertools
from collections import defaultdict

import ollama
import pandas as pd

ARGUMENTS_FILE = "argument_outputs.csv"   
OUTPUT_FILE = "judge_verdicts.csv"

JUDGE_MODELS = ["llama3.1", "gemma"]

RANDOM_SEED = 42

TEST_LIMIT = None


def load_arguments(arguments_file):
    """Return {topic: {model: argument}}, skipping any error rows."""
    df = pd.read_csv(arguments_file)
    by_topic = defaultdict(dict)
    for _, row in df.iterrows():
        argument = str(row["argument"])
        if argument.startswith("ERROR:"):
            continue
        by_topic[row["topic"]][row["model"]] = argument
    return by_topic


def build_judge_prompt(topic, first_argument, second_argument):
    return (
        f'You are judging two arguments that both support the claim: "{topic}"\n\n'
        f"Argument 1: {first_argument}\n\n"
        f"Argument 2: {second_argument}\n\n"
        "Which argument is more convincing? Respond with only "
        '"Argument 1" or "Argument 2", nothing else.'
    )


def get_verdict(judge_model, topic, first_argument, second_argument):
    """Ask judge_model to pick between the two arguments as presented (in order)."""
    prompt = build_judge_prompt(topic, first_argument, second_argument)
    response = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def parse_verdict(raw_verdict):
    """Return 1, 2, or None (unclear) from the judge's raw response."""
    text = raw_verdict.lower()
    mentions_1 = "1" in text or "argument 1" in text
    mentions_2 = "2" in text or "argument 2" in text
    if mentions_1 and not mentions_2:
        return 1
    if mentions_2 and not mentions_1:
        return 2
    return None


def main():
    random.seed(RANDOM_SEED)
    by_topic = load_arguments(ARGUMENTS_FILE)

    topics = list(by_topic.keys())
    if TEST_LIMIT is not None:
        topics = topics[:TEST_LIMIT]

    jobs = []
    for topic in topics:
        models_here = by_topic[topic]
        if len(models_here) < 2:
            print(f"Skipping '{topic}': fewer than 2 valid arguments.")
            continue
        for model_a, model_b in itertools.combinations(sorted(models_here), 2):
            for judge in JUDGE_MODELS:
                jobs.append((topic, model_a, model_b, judge))

    total = len(jobs)
    swap_flags = [True] * (total // 2) + [False] * (total - total // 2)
    random.shuffle(swap_flags)

    print(f"Running {total} judge comparisons across {len(topics)} topics...")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "judge_model",
            "topic",
            "model_a",
            "model_b",
            "swapped",
            "first_model",
            "second_model",
            "first_arg_length_words",
            "second_arg_length_words",
            "raw_verdict",
            "winner_model",
        ])

        for i, ((topic, model_a, model_b, judge), swapped) in enumerate(
            zip(jobs, swap_flags), start=1
        ):
            arg_a = by_topic[topic][model_a]
            arg_b = by_topic[topic][model_b]

            if swapped:
                first_model, first_argument = model_b, arg_b
                second_model, second_argument = model_a, arg_a
            else:
                first_model, first_argument = model_a, arg_a
                second_model, second_argument = model_b, arg_b

            print(
                f"[{i}/{total}] judge={judge} topic='{topic}' "
                f"{first_model} vs {second_model} (swapped={swapped})"
            )

            try:
                raw_verdict = get_verdict(judge, topic, first_argument, second_argument)
            except Exception as e:
                print(f"  ! judge {judge} failed: {e}")
                raw_verdict = f"ERROR: {e}"

            choice = parse_verdict(raw_verdict) if not raw_verdict.startswith("ERROR:") else None
            if choice == 1:
                winner_model = first_model
            elif choice == 2:
                winner_model = second_model
            else:
                winner_model = "UNCLEAR"

            writer.writerow([
                judge,
                topic,
                model_a,
                model_b,
                swapped,
                first_model,
                second_model,
                len(first_argument.split()),
                len(second_argument.split()),
                raw_verdict,
                winner_model,
            ])

    print(f"\nDone. Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()