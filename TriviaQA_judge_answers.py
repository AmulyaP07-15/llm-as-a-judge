import os
import re
import time
import pandas as pd
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# three models from distinct families. these have to match the models that
# generated the answers, otherwise self enhancement bias is not measurable.
JUDGE_MODELS = {
    "llama": "llama-3.1-8b-instant",
    "qwen": "qwen/qwen3.6-27b",
    "allam": "allam-2-7b"
}

INPUT = "TriviaQA_pairs.csv"
OUT = "TriviaQA_judge_verdicts.csv"
BATCH_SIZE = 50

# set to a number to test on a few pairs, None for the full run
TEST_LIMIT = None

# groq free tier is 200k tokens per day per model. stop short of the wall so
# the run exits cleanly instead of logging a stream of rate limit failures.
TPD_LIMIT = 200_000
SAFETY_MARGIN = 5_000

# 30 requests per minute is roughly one call every two seconds
MIN_INTERVAL = 2.0

tokens_used = {name: 0 for name in JUDGE_MODELS}
exhausted = set()
last_call_at = 0.0


def throttle():
    """
    sleep only for what is left of the interval rather than a flat two seconds.
    if the call itself took a second we only need to wait one more.
    """
    global last_call_at
    gap = time.time() - last_call_at
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    last_call_at = time.time()


def retry_delay_from_error(msg, attempt):
    """
    groq puts the exact wait time in the 429 body. use that rather than
    guessing, and fall back to backoff if the message format changes.
    """
    m = re.search(r"try again in (\d+)m([\d.]+)s", msg)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2)) + 2
    m = re.search(r"try again in ([\d.]+)s", msg)
    if m:
        return float(m.group(1)) + 2
    return 30 * (attempt + 1)


def format_comparison(question, first_answer, second_answer):
    """the user turn for a single comparison, same shape every time"""
    return (
        f"Context: {question}\n"
        f"Option 1: {first_answer}\n"
        f"Option 2: {second_answer}"
    )


def build_messages(question, first_answer, second_answer):
    """
    one identical prompt for all three judges so differences in verdicts come
    from the model rather than the wording.

    the example is a real user turn and assistant turn rather than text inside
    the prompt. an inline example gets treated as a pattern to continue, and
    llama was echoing the template back instead of answering. putting it in
    the message history makes clear where the model's own turn begins.

    the question is labeled Context rather than Question because a labeled
    question makes smaller models answer it instead of comparing the options.
    """
    system = (
        "You compare two candidate responses and select the better one. "
        "Never answer the context yourself. Never repeat the context or the options. "
        "Reply with exactly two lines: a Choice line and a Confidence line. "
        "Choice is 1, 2, or Neither. Confidence is a single digit 1 to 5."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": format_comparison(
            "Which planet is closest to the sun?", "Venus", "Mercury"
        )},
        {"role": "assistant", "content": "Choice: 2\nConfidence: 5"},
        {"role": "user", "content": format_comparison(
            question, first_answer, second_answer
        )}
    ]


def parse_verdict(raw):
    """
    pull a choice and confidence out of whatever the model returned.
    tries the labeled format first then falls back to looser patterns, since
    allam tends to wrap its choice inside its own phrasing.

    returns choice as 1, 2, "NEITHER" or None, plus the method used so parse
    quality can be reported and filtered on later.
    """
    if not raw:
        return None, None, "empty"

    text = raw.strip()

    # qwen wraps reasoning in think tags. drop it if the block actually closed,
    # otherwise there is no answer after it and the row is unusable.
    if "<think>" in text:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        else:
            return None, None, "truncated_think"

    # a response that repeats the prompt structure is an echo, not a verdict.
    # without this guard the loose fallbacks below will mine a number out of
    # the echoed "Option 1:" line and record a choice that was never made.
    if "context:" in text.lower() or (
        "option 1:" in text.lower() and "option 2:" in text.lower()
    ):
        return None, None, "echoed_prompt"

    choice = None
    confidence = None
    method = None

    # first preference, the format the prompt asked for
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("choice:"):
            candidate = line.split(":", 1)[-1].strip().upper()
            if candidate in ("1", "2"):
                choice = int(candidate)
                method = "labeled"
            elif candidate == "NEITHER":
                choice = "NEITHER"
                method = "labeled"
        elif line.lower().startswith("confidence:"):
            digits = "".join(ch for ch in line if ch.isdigit())
            if digits:
                val = int(digits[0])
                # discard anything outside the scale rather than storing junk
                confidence = val if 1 <= val <= 5 else None

    upper = text.upper()

    # bare reply, just the number or the word and nothing else
    if choice is None:
        if upper in ("1", "2"):
            choice = int(upper)
            method = "bare"
        elif upper == "NEITHER":
            choice = "NEITHER"
            method = "bare"

    # phrasing like "Option 2 is stronger". anchored on the word so a stray
    # digit elsewhere in the sentence does not trigger a false match.
    if choice is None:
        m = re.search(r"\bOPTION\s+([12])\b", upper)
        if m:
            choice = int(m.group(1))
            method = "option_phrase"

    # allam sometimes writes the content then the number, like "PORTUGAL (1)"
    if choice is None:
        m = re.search(r"\(([12])\)", upper)
        if m:
            choice = int(m.group(1))
            method = "paren_number"

    if choice is None:
        return None, confidence, "unparsed"

    return choice, confidence, method


def call_judge(judge_name, judge_id, messages):
    """
    one call with rate limit retry and token accounting. a 429 gets retried
    rather than written as a null verdict, otherwise the results fill up with
    failures that never actually happened.
    """
    kwargs = {
        "model": judge_id,
        "messages": messages,
        # the reply is two short lines, so this is generous. keeping it tight
        # stops a rambling model from eating the daily budget on one call.
        "max_tokens": 24,
        "temperature": 0
    }

    # qwen reasons before answering by default, which returns a truncated think
    # block and burns roughly 2500 tokens a call instead of under 40.
    if judge_name == "qwen":
        kwargs["reasoning_effort"] = "none"

    for attempt in range(3):
        throttle()
        try:
            start = time.time()
            response = client.chat.completions.create(**kwargs)
            elapsed = round(time.time() - start, 2)

            if getattr(response, "usage", None):
                tokens_used[judge_name] += response.usage.total_tokens

            return response.choices[0].message.content.strip(), elapsed, None

        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg or "429" in msg:
                # a daily limit will not clear inside this run, so stop calling
                # this model rather than retrying into the same wall
                if "per day" in msg or "TPD" in msg:
                    exhausted.add(judge_name)
                    return None, None, "daily_limit_reached"
                wait = retry_delay_from_error(msg, attempt)
                print(f"  rate limited on {judge_name}, waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            return None, None, str(e)

    return None, None, "rate_limited_after_retries"


def score_correctness(winning_answer, correct_answer):
    """
    ground truth check, objective task only. loose substring match because
    model answers are free text and TriviaQA only gave us the first alias,
    so this undercounts slightly rather than overcounting.
    """
    if winning_answer is None:
        return None

    winning = str(winning_answer).lower().strip()
    correct = str(correct_answer).lower().strip()

    # require a few characters so short strings do not match by accident
    if len(correct) < 3 or len(winning) < 3:
        return correct == winning

    return correct in winning or winning in correct


df = pd.read_csv(INPUT)
if TEST_LIMIT is not None:
    df = df.head(TEST_LIMIT)
    print(f"test mode, {TEST_LIMIT} pairs only")

# resume from checkpoint so an interrupted run or an exhausted daily budget
# can be picked up later without redoing completed work
if os.path.exists(OUT):
    df_existing = pd.read_csv(OUT)
    done = set(zip(
        df_existing["question"],
        df_existing["model_a"],
        df_existing["model_b"],
        df_existing["judge_model"]
    ))
    results = df_existing.to_dict("records")
    print(f"resuming from {len(results)} already done")
else:
    done = set()
    results = []

for _, row in df.iterrows():
    messages = build_messages(
        row["question"],
        row["first_answer"],
        row["second_answer"]
    )

    # every pair gets judged by all three models
    for judge_name, judge_id in JUDGE_MODELS.items():

        key = (row["question"], row["model_a"], row["model_b"], judge_name)
        if key in done:
            continue

        # once a model is out of daily budget, skip it instead of writing
        # fake failures. the resume logic picks it up tomorrow.
        if judge_name in exhausted:
            continue

        if tokens_used[judge_name] > TPD_LIMIT - SAFETY_MARGIN:
            print(f"  {judge_name} near daily limit, skipping its remaining calls")
            exhausted.add(judge_name)
            continue

        print(f"judging {row['first_model']} vs {row['second_model']} with {judge_name}...")

        raw, elapsed, error = call_judge(judge_name, judge_id, messages)

        if error:
            print(f"  error: {error}")
            choice, confidence, method = None, None, "api_error"
        else:
            choice, confidence, method = parse_verdict(raw)

        # resolve the choice back to a model name. a failed parse and a genuine
        # abstain are kept separate so the neither rate stays honest.
        if choice == 1:
            winner_model = row["first_model"]
            winning_answer = row["first_answer"]
        elif choice == 2:
            winner_model = row["second_model"]
            winning_answer = row["second_answer"]
        elif choice == "NEITHER":
            winner_model = "NEITHER"
            winning_answer = None
        else:
            winner_model = None
            winning_answer = None

        print(f"  choice: {choice}, winner: {winner_model}, confidence: {confidence}, parse: {method}")

        results.append({
            "judge_model": judge_name,
            "question": row["question"],
            "correct_answer": row["correct_answer"],
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "swapped": row["swapped"],
            "first_model": row["first_model"],
            "second_model": row["second_model"],
            "first_answer": row["first_answer"],
            "second_answer": row["second_answer"],
            "first_length_words": row["first_length_words"],
            "second_length_words": row["second_length_words"],
            "raw_verdict": raw,
            "choice": choice,
            "winner_model": winner_model,
            "confidence": confidence,
            "correct_match": score_correctness(winning_answer, row["correct_answer"]),
            "parse_method": method,
            "response_time": elapsed
        })

        if len(results) % BATCH_SIZE == 0:
            pd.DataFrame(results).to_csv(OUT, index=False)
            print(f"checkpoint saved at {len(results)} judgments")

pd.DataFrame(results).to_csv(OUT, index=False)
print(f"\ndone. saved {len(results)} judgments to {OUT}")

for name in JUDGE_MODELS:
    print(f"{name}: {tokens_used[name]:,} tokens")

if exhausted:
    print(f"\nhit daily limits: {', '.join(sorted(exhausted))}")
    print("rerun tomorrow, resume logic will continue from here")