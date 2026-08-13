import csv
import os
import random
import re
import itertools
from collections import defaultdict

import pandas as pd

from groq_utils import call_groq, MODEL_IDS

ARGUMENTS_FILE = "argument_outputs.csv"   
OUTPUT_FILE = "judge_verdicts.csv"

JUDGE_MODELS = list(MODEL_IDS.keys())  

RANDOM_SEED = 42

TEST_LIMIT = None

ECHO_MARKERS = ["<argument 1", "argument 1, argument 2, or neither"]


def load_arguments(arguments_file):
    """
    loads generated arguments, skipping any error rows.

    input: argument_outputs.csv path
    output: {topic: {model: argument}}
    """
    df = pd.read_csv(arguments_file)
    by_topic = defaultdict(dict)
    for _, row in df.iterrows():
        argument = str(row["argument"])
        if argument.startswith("ERROR:"):
            continue
        by_topic[row["topic"]][row["model"]] = argument
    return by_topic


def load_completed_keys(output_file):
    """
    finds which comparisons are already logged, so a resumed run can skip them.

    input: output csv path
    output: set of (topic, model_a, model_b, judge) already logged
    """
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
    except pd.errors.EmptyDataError:
        return set()
    return set(zip(df["topic"], df["model_a"], df["model_b"], df["judge_model"]))


def build_judge_prompt(topic, first_argument, second_argument):
    """
    builds the prompt asking a judge to pick the stronger argument.

    input: topic, both arguments
    output: the prompt string sent to the judge
    """
    return (
        f'You are judging two arguments that both support the claim: "{topic}"\n\n'
        f"Argument 1: {first_argument}\n\n"
        f"Argument 2: {second_argument}\n\n"
        "Which argument is more convincing? If neither is clearly better, say Neither.\n\n"
        "Respond in exactly this format and nothing else:\n"
        "Winner: <Argument 1, Argument 2, or Neither>\n"
        "Confidence: <a number from 1 to 5, where 5 is extremely confident>"
    )


def is_echoed_prompt(raw_text):
    """
    checks whether the model just echoed the prompt's placeholder text back instead of answering.

    input: raw model output
    output: True if it looks like an echo
    """
    text = raw_text.lower()
    return any(marker in text for marker in ECHO_MARKERS)


def parse_verdict(raw_text):
    """
    pulls the winner and confidence out of the judge's raw response.

    input: raw judge text
    output: (choice, confidence) -- choice is 1, 2, "NEITHER", or None; confidence is 1-5 or None
    """
    winner_match = re.search(r"winner:\s*(.+)", raw_text, re.IGNORECASE)
    # \d+(?:\.\d+)? captures decimals too (e.g. "4.5"), not just whole numbers.
    confidence_match = re.search(r"confidence:\s*(\d+(?:\.\d+)?)", raw_text, re.IGNORECASE)

    confidence = None
    if confidence_match:
        value = float(confidence_match.group(1))
        # The prompt asks for a 1-5 scale. Some models (allam in particular)
        # answer on a different scale (e.g. "9 out of 10", "Confidence: 10").
        # A value outside 1-5 isn't on the same scale as the rest of the
        # data, so keep it as a missing value rather than silently mixing
        # scales in downstream aggregates.
        if 1 <= value <= 5:
            confidence = value

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
    """
    asks judge_model to pick between the two arguments as presented.

    input: judge model, topic, both arguments in display order
    output: (raw response text, seconds elapsed)
    """
    prompt = build_judge_prompt(topic, first_argument, second_argument)
    messages = [{"role": "user", "content": prompt}]
    return call_groq(judge_model, messages, max_tokens=50)  # (raw_text, elapsed_sec)


def build_jobs(by_topic, topics):
    """
    builds the full list of judge comparisons. the swap decision is made once per pair,
    not per judge, so every judge sees the same argument order for a given pair.

    input: {topic: {model: argument}}, topic list
    output: list of (topic, model_a, model_b, judge, swapped) jobs
    """
    pairs = []
    for topic in topics:
        models_here = by_topic[topic]
        if len(models_here) < 2:
            print(f"Skipping '{topic}': fewer than 2 valid arguments.")
            continue
        for model_a, model_b in itertools.combinations(sorted(models_here), 2):
            pairs.append((topic, model_a, model_b))

    total_pairs = len(pairs)
    swap_flags = [True] * (total_pairs // 2) + [False] * (total_pairs - total_pairs // 2)
    random.shuffle(swap_flags)

    jobs = []
    for (topic, model_a, model_b), swapped in zip(pairs, swap_flags):
        for judge in JUDGE_MODELS:
            jobs.append((topic, model_a, model_b, judge, swapped))
    return jobs


def main():
    """
    runs every judge comparison and writes the results.

    input: nothing
    output: nothing, writes judge_verdicts.csv
    """
    random.seed(RANDOM_SEED)
    by_topic = load_arguments(ARGUMENTS_FILE)

    topics = list(by_topic.keys())
    if TEST_LIMIT is not None:
        print(f"*** TEST_LIMIT={TEST_LIMIT} is set -- this is a partial test run, "
              f"not the full dataset. ***")
        topics = topics[:TEST_LIMIT]

    jobs = build_jobs(by_topic, topics)
    total = len(jobs)

    completed = load_completed_keys(OUTPUT_FILE)
    resuming = len(completed) > 0
    if resuming:
        print(f"Found {len(completed)} completed rows in {OUTPUT_FILE} -- resuming.")

    print(f"{total} total judge comparisons across {len(topics)} topics "
          f"({total - len(completed)} remaining).")

    mode = "a" if resuming else "w"
    with open(OUTPUT_FILE, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not resuming:
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

        for i, (topic, model_a, model_b, judge, swapped) in enumerate(jobs, start=1):
            key = (topic, model_a, model_b, judge)
            if key in completed:
                continue

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
                if is_echoed_prompt(raw_response):
                    winner_model = "ECHOED_PROMPT"
                    confidence = None
                else:
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
            f.flush()  # persist progress row-by-row so a crash doesn't lose the run

    print(f"\nDone. Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()