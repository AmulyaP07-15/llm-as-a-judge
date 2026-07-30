import itertools

import pandas as pd
from scipy.stats import binomtest, chi2_contingency
from sklearn.metrics import cohen_kappa_score

VERDICTS_FILE = "TriviaQA_judge_verdicts.csv"
SAMPLED_FILE = "TriviaQA_sampled.csv"

# a verdict is decisive when the judge actually picked one of the two answers
NON_DECISIVE = {"NEITHER", None}

# aliases shorter than this are ignored when matching, they produce
# accidental hits on common short strings
MIN_ALIAS_LEN = 3


def normalise(text):
    """
    strip punctuation so "See-saw" still matches the alias "see saw".
    apostrophes get deleted rather than spaced, so mcdonald's and mcdonalds
    end up the same.

    input: any string or None
    output: lowercase string, punctuation flattened to spaces
    """
    if text is None:
        return ""

    out = []
    for ch in str(text).lower():
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        elif ch in "'\u2019":
            # apostrophes are dropped rather than spaced, so that mcdonald's
            # and mcdonalds collapse to the same string
            continue
        else:
            out.append(" ")

    return " ".join("".join(out).split())


def load_data(verdicts_file=VERDICTS_FILE, sampled_file=SAMPLED_FILE):
    """
    load the verdicts and build the question to aliases lookup.

    aliases come from the sampled file since the verdicts only carry one
    answer string, and one alias is not enough to score free text fairly.

    input: paths to the verdicts csv and the sampled questions csv
    output: verdicts dataframe, and a dict of question to alias set
    """
    df = pd.read_csv(verdicts_file)

    # choice comes back as a string because NEITHER shares the column
    df["choice"] = df["choice"].astype(str)

    sampled = pd.read_csv(sampled_file)
    alias_map = {}
    for _, row in sampled.iterrows():
        raw = row.get("answer_aliases")
        if pd.isna(raw):
            aliases = [str(row["answer_text"])]
        else:
            aliases = str(raw).split("|")
        cleaned = {normalise(a) for a in aliases}
        alias_map[row["question"]] = {a for a in cleaned if a}

    return df, alias_map


def decisive(df):
    """
    keep only verdicts where a model actually won. abstains and parse
    failures both go, since neither says anything about what a judge preferred.

    input: full verdicts dataframe
    output: copy with only rows that have a real winner
    """
    return df[~df["winner_model"].isin(NON_DECISIVE) & df["winner_model"].notna()].copy()


def winning_answer(row):
    """
    get the text of whichever answer won.

    input: one verdicts row
    output: the winning answer string, or None if the judge abstained
    """
    if row["choice"] == "1":
        return row["first_answer"]
    if row["choice"] == "2":
        return row["second_answer"]
    return None


def is_correct(answer, aliases):
    """
    check a free text answer against every acceptable form of the truth.

    matching only goes one way. an alias can sit inside the model's answer,
    since models pad with extra words, but not the reverse. otherwise a one
    word answer like "rat" matches the alias "yolanda rat" and scores correct
    when it clearly is not. exact matches are always fine.

    input: an answer string or None, and a set of aliases
    output: True, False, or None if there was nothing to score
    """
    if answer is None or not aliases:
        return None

    text = normalise(answer)
    if not text:
        return None

    for alias in aliases:
        if alias == text:
            return True
        if len(alias) >= MIN_ALIAS_LEN and alias in text:
            return True

    return False


def add_correctness(df, alias_map):
    """
    score every row against the aliases and flag the ties.

    a tie is where both models gave the same answer once punctuation and case
    are normalised. it happens a lot on short factual questions.

    input: verdicts dataframe, and the question to aliases lookup
    output: copy with winning_answer, correct, and is_tie columns
    """
    df = df.copy()
    df["winning_answer"] = df.apply(winning_answer, axis=1)
    df["correct"] = df.apply(
        lambda r: is_correct(r["winning_answer"], alias_map.get(r["question"], set())),
        axis=1
    )
    df["is_tie"] = df.apply(
        lambda r: normalise(r["first_answer"]) == normalise(r["second_answer"]),
        axis=1
    )

    # both answers get scored, not just the winner, so each comparison can be
    # classified by how much room the judge had to be right
    df["first_correct"] = df.apply(
        lambda r: is_correct(r["first_answer"], alias_map.get(r["question"], set())),
        axis=1
    )
    df["second_correct"] = df.apply(
        lambda r: is_correct(r["second_answer"], alias_map.get(r["question"], set())),
        axis=1
    )

    def classify(r):
        fc, sc = r["first_correct"], r["second_correct"]
        if fc is None or sc is None:
            return None
        if fc and sc:
            return "both right"
        if not fc and not sc:
            return "both wrong"
        return "one right"

    df["case"] = df.apply(classify, axis=1)
    return df


def tie_summary(df):
    """
    how often the two models gave the same answer, and what judges did then.

    a tie has no quality difference to detect, so a judge can only fall back on
    position. that makes ties useless for measuring preference but a clean
    controlled test of position bias, quality held exactly constant.

    every other metric runs on non ties only, which is the usual convention.

    input: decisive verdicts, after add_correctness
    output: a single row dataframe with the tie rate and the first position
            rate within ties and within genuine comparisons
    """
    ties = df[df["is_tie"]]
    real = df[~df["is_tie"]]

    return pd.DataFrame([{
        "tie_rate": df["is_tie"].mean(),
        "n_ties": len(ties),
        "n_genuine": len(real),
        "first_pos_rate_on_ties": (ties["choice"] == "1").mean() if len(ties) else float("nan"),
        "first_pos_rate_on_genuine": (real["choice"] == "1").mean() if len(real) else float("nan"),
    }])


def outcome_rates(df_all):
    """
    how often each judge abstained or gave something unparseable.

    kept separate on purpose. counting a failed call as an abstain would make
    a judge look more indecisive than it actually was.

    input: full verdicts dataframe
    output: one row per judge with n, neither_rate, unparsed_rate
    """
    rows = []
    for judge, group in df_all.groupby("judge_model"):
        n = len(group)
        rows.append({
            "judge": judge,
            "n": n,
            "neither_rate": (group["winner_model"] == "NEITHER").mean(),
            "unparsed_rate": group["winner_model"].isna().mean(),
        })
    return pd.DataFrame(rows)


def self_enhancement(df):
    """
    does a model favour its own answers when it is the judge.

    compares how often a model wins while judging itself against how often it
    wins on those same pairs when the third model judges.

    the baseline is the third model only, not everyone who is not the judge.
    in a pair of A and B both have a stake, so only C is genuinely neutral.
    pooling them would mix a neutral rater in with an interested one.

    input: decisive verdicts
    output: one row per model with self_rate, neutral_rate, the difference,
            and a chi square p value
    """
    models = sorted(set(df["model_a"]) | set(df["model_b"]))
    rows = []

    for model in models:
        in_pair = df[(df["model_a"] == model) | (df["model_b"] == model)]

        self_cases = in_pair[in_pair["judge_model"] == model]
        # the third model, neither competitor
        neutral_cases = in_pair[
            (in_pair["judge_model"] != in_pair["model_a"])
            & (in_pair["judge_model"] != in_pair["model_b"])
        ]

        self_wins = int((self_cases["winner_model"] == model).sum())
        self_n = len(self_cases)
        neutral_wins = int((neutral_cases["winner_model"] == model).sum())
        neutral_n = len(neutral_cases)

        if self_n == 0 or neutral_n == 0:
            continue

        table = [
            [self_wins, self_n - self_wins],
            [neutral_wins, neutral_n - neutral_wins],
        ]

        # chi square needs every cell populated, fall back when a row is degenerate
        if min(min(r) for r in table) < 0 or self_n < 2 or neutral_n < 2:
            p = float("nan")
        else:
            try:
                _, p, _, _ = chi2_contingency(table)
            except ValueError:
                p = float("nan")

        rows.append({
            "model": model,
            "self_rate": self_wins / self_n,
            "self_n": self_n,
            "neutral_rate": neutral_wins / neutral_n,
            "neutral_n": neutral_n,
            "difference": self_wins / self_n - neutral_wins / neutral_n,
            "p_value": p,
        })

    return pd.DataFrame(rows)


def self_enhancement_by_position(df):
    """
    the same comparison, but holding presentation order fixed.

    position bias is strong enough here that it can swamp or fake a self
    enhancement effect. splitting by whether the model's own answer was shown
    first or second separates the two. a model that beats the neutral judge in
    both positions is genuinely favouring itself. one that only beats it in a
    single position is showing position bias in disguise.

    input: decisive verdicts, ties already removed
    output: two rows per model, one per position, each with self and neutral
            rates, the difference, and a chi square p value
    """
    models = sorted(set(df["model_a"]) | set(df["model_b"]))
    rows = []

    for model in models:
        in_pair = df[(df["model_a"] == model) | (df["model_b"] == model)]

        for shown_first in (True, False):
            sub = in_pair[(in_pair["first_model"] == model) == shown_first]

            self_cases = sub[sub["judge_model"] == model]
            neutral_cases = sub[
                (sub["judge_model"] != sub["model_a"])
                & (sub["judge_model"] != sub["model_b"])
            ]

            self_n, neutral_n = len(self_cases), len(neutral_cases)
            if self_n < 2 or neutral_n < 2:
                continue

            self_wins = int((self_cases["winner_model"] == model).sum())
            neutral_wins = int((neutral_cases["winner_model"] == model).sum())

            table = [
                [self_wins, self_n - self_wins],
                [neutral_wins, neutral_n - neutral_wins],
            ]
            try:
                _, p, _, _ = chi2_contingency(table)
            except ValueError:
                p = float("nan")

            rows.append({
                "model": model,
                "position": "first" if shown_first else "second",
                "self_rate": self_wins / self_n,
                "self_n": self_n,
                "neutral_rate": neutral_wins / neutral_n,
                "neutral_n": neutral_n,
                "difference": self_wins / self_n - neutral_wins / neutral_n,
                "p_value": p,
            })

    return pd.DataFrame(rows)


def first_position_preference(df):
    """
    how often judges picked whatever was shown first.

    order was randomised upstream in an exact half split, so an unbiased judge
    should sit near 0.5.

    this is not a paired flip rate. each judge only ever sees a pair in one
    order, so we cannot watch a verdict flip on reversal. doing that would
    double the judge calls and the token budget will not cover it.

    input: decisive verdicts
    output: a row for ALL and one per judge, with rate, n, and a binomial p
    """
    rows = []

    n = len(df)
    wins = int((df["choice"] == "1").sum())
    rows.append({
        "judge": "ALL",
        "rate": wins / n if n else float("nan"),
        "n": n,
        "p_value": binomtest(wins, n, 0.5).pvalue if n else float("nan"),
    })

    for judge, group in df.groupby("judge_model"):
        gn = len(group)
        gwins = int((group["choice"] == "1").sum())
        rows.append({
            "judge": judge,
            "rate": gwins / gn if gn else float("nan"),
            "n": gn,
            "p_value": binomtest(gwins, gn, 0.5).pvalue if gn else float("nan"),
        })

    return pd.DataFrame(rows)


def verbosity_preference(df):
    """
    do judges favour the longer answer.

    one row per comparison, just asking whether the longer one won. going per
    answer instead would add two rows for every comparison, a win and a loss
    by construction, and those are not independent so a correlation test on
    them would not hold up.

    equal length pairs carry no signal so they get dropped.

    input: decisive verdicts
    output: a row for ALL and one per judge, with longer_wins_rate, n, and a
            binomial p
    """
    rows = []

    def compute(group, label):
        usable = group[group["first_length_words"] != group["second_length_words"]]
        n = len(usable)
        if n == 0:
            return {"judge": label, "longer_wins_rate": float("nan"),
                    "n": 0, "p_value": float("nan")}

        longer_is_first = usable["first_length_words"] > usable["second_length_words"]
        picked_first = usable["choice"] == "1"
        longer_won = int((longer_is_first == picked_first).sum())

        return {
            "judge": label,
            "longer_wins_rate": longer_won / n,
            "n": n,
            "p_value": binomtest(longer_won, n, 0.5).pvalue,
        }

    rows.append(compute(df, "ALL"))
    for judge, group in df.groupby("judge_model"):
        rows.append(compute(group, judge))

    return pd.DataFrame(rows)


def verbosity_by_position(df):
    """
    does the longer answer win, holding presentation order fixed.

    pooling the two positions hides the confound. if a judge has a strong first
    position preference, then whenever the longer answer happens to sit first it
    wins, and that shows up as verbosity bias when it is really position again.

    splitting separates them. a judge that favours length in both positions is
    genuinely responding to length. one that only shows it in a single position
    is showing position bias wearing a different hat.

    input: decisive verdicts, ties already removed
    output: two rows per judge, one per position of the longer answer, with
            longer_wins_rate, n, and a binomial p against 0.5
    """
    usable = df[df["first_length_words"] != df["second_length_words"]]
    rows = []

    for judge, group in usable.groupby("judge_model"):
        for longer_first in (True, False):
            sub = group[(group["first_length_words"] > group["second_length_words"]) == longer_first]
            n = len(sub)
            if n == 0:
                continue

            picked_first = sub["choice"] == "1"
            longer_won = int((picked_first == longer_first).sum())

            rows.append({
                "judge": judge,
                "longer_shown": "first" if longer_first else "second",
                "longer_wins_rate": longer_won / n,
                "n": n,
                "p_value": binomtest(longer_won, n, 0.5).pvalue,
            })

    return pd.DataFrame(rows)


def ground_truth_accuracy(df):
    """
    how often the picked answer was actually right.

    this is what separates real quality judgment from a judge just reacting to
    position or length. objective task only, since argument quality has no
    ground truth.

    input: decisive verdicts, after add_correctness
    output: a row for ALL and one per judge, with accuracy and n
    """
    rows = []
    scored = df[df["correct"].notna()]

    rows.append({
        "judge": "ALL",
        "accuracy": scored["correct"].mean(),
        "n": len(scored),
    })
    for judge, group in scored.groupby("judge_model"):
        rows.append({
            "judge": judge,
            "accuracy": group["correct"].mean(),
            "n": len(group),
        })

    return pd.DataFrame(rows)


def discrimination_accuracy(df):
    """
    how often a judge picked the right answer when only one was right.

    plain accuracy is misleading here. a comparison where both answers are
    wrong scores zero whatever the judge does, and one where both are right
    scores one regardless. neither says anything about the judge. pooling all
    three cases mostly measures how good the underlying models were, not how
    well the judge told them apart.

    restricting to comparisons with exactly one correct answer isolates the
    judge's actual discrimination, and 0.5 is the meaningful null.

    input: decisive verdicts, after add_correctness
    output: a row for ALL and one per judge, with accuracy on the one right
            subset, n, and a binomial p against 0.5
    """
    one_right = df[df["case"] == "one right"]
    rows = []

    def compute(group, label):
        n = len(group)
        if n == 0:
            return {"judge": label, "accuracy": float("nan"),
                    "n": 0, "p_value": float("nan")}
        hits = int(group["correct"].sum())
        return {
            "judge": label,
            "accuracy": hits / n,
            "n": n,
            "p_value": binomtest(hits, n, 0.5).pvalue,
        }

    rows.append(compute(one_right, "ALL"))
    for judge, group in one_right.groupby("judge_model"):
        rows.append(compute(group, judge))

    return pd.DataFrame(rows)


def case_breakdown(df):
    """
    how the comparisons split by how many answers were correct.

    context for the accuracy numbers. a large both wrong bucket drags plain
    accuracy down without reflecting anything about the judges.

    input: decisive verdicts, after add_correctness
    output: one row per case with the count and share
    """
    counts = df["case"].value_counts(dropna=True)
    total = counts.sum()
    return pd.DataFrame([
        {"case": case, "n": int(n), "share": n / total}
        for case, n in counts.items()
    ])


def confidence_summary(df):
    """
    what confidence each judge reported, split by whether it was right.

    a judge just as confident when wrong as when right is not tracking its own
    reliability, which matters if we ever weight verdicts by confidence.

    input: decisive verdicts, after add_correctness
    output: one row per judge with mean confidence overall, when correct, and
            when wrong
    """
    rows = []
    for judge, group in df.groupby("judge_model"):
        scored = group[group["correct"].notna()]
        rows.append({
            "judge": judge,
            "mean_confidence": group["confidence"].mean(),
            "confidence_when_correct": scored.loc[scored["correct"], "confidence"].mean(),
            "confidence_when_wrong": scored.loc[~scored["correct"].astype(bool), "confidence"].mean(),
            "n": len(group),
        })
    return pd.DataFrame(rows)


def response_time_summary(df_all):
    """
    how long each judge took per call. median sits next to the mean because a
    few slow calls can drag the mean around.

    input: full verdicts dataframe
    output: one row per judge with mean and median seconds
    """
    rows = []
    for judge, group in df_all.groupby("judge_model"):
        rows.append({
            "judge": judge,
            "mean_seconds": group["response_time"].mean(),
            "median_seconds": group["response_time"].median(),
            "n": group["response_time"].notna().sum(),
        })
    return pd.DataFrame(rows)


def inter_judge_agreement(df_all):
    """
    cohen's kappa between each pair of judges.

    the item key is question plus the two competing models. order stays out of
    it deliberately, since all three judges saw a pair the same way and the
    winner is already a model name. putting order in would split each item and
    roughly halve the sample.

    runs on all verdicts, not just decisive ones, so two judges both abstaining
    counts as agreeing.

    input: full verdicts dataframe
    output: one row per judge pair with kappa, shared items, raw agreement
    """
    by_item = {}
    for _, row in df_all.iterrows():
        key = (row["question"], row["model_a"], row["model_b"])
        label = row["winner_model"] if pd.notna(row["winner_model"]) else "UNPARSED"
        by_item.setdefault(key, {})[row["judge_model"]] = label

    judges = sorted(df_all["judge_model"].unique())
    rows = []

    for jx, jy in itertools.combinations(judges, 2):
        lx, ly = [], []
        for verdicts in by_item.values():
            if jx in verdicts and jy in verdicts:
                lx.append(verdicts[jx])
                ly.append(verdicts[jy])

        if len(lx) >= 2 and len(set(lx + ly)) >= 2:
            kappa = cohen_kappa_score(lx, ly)
        else:
            kappa = float("nan")

        rows.append({
            "pair": f"{jx} vs {jy}",
            "kappa": kappa,
            "n_shared": len(lx),
            "raw_agreement": sum(a == b for a, b in zip(lx, ly)) / len(lx) if lx else float("nan"),
        })

    return pd.DataFrame(rows)


def show(title, frame):
    """
    print one results table under a heading.

    input: a title and a dataframe
    output: nothing, prints
    """
    print(f"\n{title}")
    print("-" * len(title))
    print(frame.to_string(index=False))


def main():
    """
    run every metric and print the tables.

    correctness gets recomputed here rather than read from the correct_match
    column, since that column was written against a single alias and marks
    plenty of right answers wrong.

    input: nothing, reads the two csvs from the working directory
    output: nothing, prints
    """
    df_all, alias_map = load_data()
    df_all = add_correctness(df_all, alias_map)
    df = decisive(df_all)

    # ties carry no quality signal, so every preference metric runs without
    # them. they are reported separately since they double as a clean position
    # bias test with quality held constant.
    genuine = df[~df["is_tie"]]
    genuine_all = df_all[~df_all["is_tie"]]

    print(f"{len(df_all)} verdicts, {len(df)} decisive, {len(genuine)} genuine")
    print(f"{df_all['question'].nunique()} questions, "
          f"{df_all['judge_model'].nunique()} judges")

    show("Ties", tie_summary(df))
    show("Outcome rates", outcome_rates(genuine_all))
    show("Self enhancement", self_enhancement(genuine))
    show("Self enhancement by position", self_enhancement_by_position(genuine))
    show("First position preference", first_position_preference(genuine))
    show("Verbosity preference", verbosity_preference(genuine))
    show("Verbosity by position", verbosity_by_position(genuine))
    show("Comparison types", case_breakdown(genuine))
    show("Ground truth accuracy (all cases)", ground_truth_accuracy(genuine))
    show("Discrimination accuracy (one right only)", discrimination_accuracy(genuine))
    show("Judge confidence", confidence_summary(genuine))
    show("Response time", response_time_summary(df_all))
    show("Inter judge agreement", inter_judge_agreement(genuine_all))


if __name__ == "__main__":
    main()