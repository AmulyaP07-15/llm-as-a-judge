import csv
import random
import re
import time
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
        "Which argument is more convincing? If neither is clearly better, say Neither.\n\n"
        "Respond in exactly this format and nothing else:\n"
        "Winner: <Argument 1, Argument 2, or Neither>\n"
        "Confidence: <a number from 1 to 5, where 5 is extremely confident>"
    )


def parse_verdict(raw_text):
    """
    Returns (choice, confidence).
    choice is 1, 2, "NEITHER", or None (couldn't be parsed).

    Matches on the phrases "argument 1" / "argument 2" rather than bare
    digits, so a response like "Argument 2 is stronger than Argument 1"
    doesn't trip both checks and collapse to None.
    """
    winner_match = re.search(r"winner:\s*(.+)", raw_text, re.IGNORECASE)
    confidence_match = re.search(r"confidence:\s*(\d+)", raw_text, re.IGNORECASE)

    confidence = int(confidence_match.group(1)) if confidence_match else None

    if not winner_match:
        return None, confidence

    winner_text = winner_match.group(1).strip().lower()
    if "neither" in winner_text:
        return "NEITHER", confidence
    if "argument 1" in winner_text:
        return 1, confidence
    if "argument 2" in winner_text:
        return 2, confidence
    return None, confidence


def get_verdict(judge_model, topic, first_argument, second_argument):
    """Ask judge_model to pick between the two arguments as presented (in order)."""
    prompt = build_judge_prompt(topic, first_argument, second_argument)
    start = time.time()
    response = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - start
    return response["message"]["content"].strip(), elapsed


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
            "response_time_sec",
            "confidence",
            "raw_response",
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
                raw_response, elapsed = get_verdict(
                    judge, topic, first_argument, second_argument
                )
                choice, confidence = parse_verdict(raw_response)
                if choice == 1:
                    winner_model = first_model
                elif choice == 2:
                    winner_model = second_model
                elif choice == "NEITHER":
                    winner_model = "NEITHER"
                else:
                    winner_model = "UNCLEAR"
            except Exception as e:
                print(f"  ! judge {judge} failed: {e}")
                raw_response = f"ERROR: {e}"
                elapsed = None
                confidence = None
                winner_model = "API_ERROR"

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
                elapsed,
                confidence,
                raw_response,
                winner_model,
            ])

    print(f"\nDone. Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()