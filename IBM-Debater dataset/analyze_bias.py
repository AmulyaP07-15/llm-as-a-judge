import itertools
from collections import defaultdict

import pandas as pd
from scipy.stats import binomtest, chi2_contingency
from sklearn.metrics import cohen_kappa_score

VERDICTS_FILE = "judge_verdicts.csv"
ARGUMENTS_FILE = "argument_outputs.csv"
NON_DECISIVE = {"NEITHER", "UNCLEAR", "API_ERROR", "ECHOED_PROMPT"}


def load_all(path=VERDICTS_FILE):
    return pd.read_csv(path)


def load_argument_texts(path=ARGUMENTS_FILE):
    """{(topic, model): argument_text} used for the content-tie check."""
    df = pd.read_csv(path)
    return {
        (row["topic"], row["model"]): str(row["argument"]).strip()
        for _, row in df.iterrows()
    }


def find_content_tied_pairs(df, argument_texts):
    """
    Returns the set of (topic, model_a, model_b) whose two arguments are
    textually identical (exact match after stripping whitespace). These
    carry zero information for any bias question -- there's nothing to
    differentiate -- so they should be confirmed and excluded rather than
    assumed not to exist.
    """
    tied = set()
    pairs_seen = set(zip(df["topic"], df["model_a"], df["model_b"]))
    for topic, model_a, model_b in pairs_seen:
        arg_a = argument_texts.get((topic, model_a))
        arg_b = argument_texts.get((topic, model_b))
        if arg_a is not None and arg_b is not None and arg_a == arg_b:
            tied.add((topic, model_a, model_b))
    return tied


def drop_content_tied(df, tied_pairs):
    if not tied_pairs:
        return df
    mask = ~df.apply(
        lambda r: (r["topic"], r["model_a"], r["model_b"]) in tied_pairs, axis=1
    )
    return df[mask].copy()


def load_decisive(df):
    """Rows with an actual winning model -- excludes NEITHER/UNCLEAR/API_ERROR/ECHOED_PROMPT."""
    return df[~df["winner_model"].isin(NON_DECISIVE)].copy()


def outcome_rates(df):
    """Rate AND count of each non-decisive outcome, so a 0.000 rate is
    visibly a genuine zero rather than the outcome never being logged."""
    total = len(df)
    result = {}
    for outcome in NON_DECISIVE:
        count = int((df["winner_model"] == outcome).sum())
        rate = count / total if total else float("nan")
        result[outcome] = {"rate": rate, "count": count}
    return result


def _chi_square(wins_a, n_a, wins_b, n_b):
    if not n_a or not n_b:
        return float("nan"), float("nan")
    table = [[wins_a, n_a - wins_a], [wins_b, n_b - wins_b]]
    try:
        chi2, p_value, _, _ = chi2_contingency(table)
        return chi2, p_value
    except ValueError:
        # a row/column sums to zero (all wins or all losses) -- chi-square
        # isn't well-defined here
        return float("nan"), float("nan")


def self_enhancement_rate(df):
    """
    Per model: self rate (wins when the model judges itself) vs. baseline
    rate (wins when a neutral third-party model judges it), each ALSO
    split by whether the model's own argument was shown first or second --
    self and baseline are split the same way, so each position row is a
    genuine self-vs-neutral comparison, not just the self rate on its own.

    The position split matters here specifically: if a model only shows
    self-preference when its argument happens to sit in position one (or
    two), pooling across positions could make a real, position-dependent
    effect look weaker than it is, or make two opposite-signed position
    effects cancel into something that looks like no bias at all.
    """
    is_self_case = (df["judge_model"] == df["model_a"]) | (df["judge_model"] == df["model_b"])
    self_cases = df[is_self_case]
    neutral_cases = df[~is_self_case]

    models = sorted(pd.concat([df["model_a"], df["model_b"]]).unique())
    per_model = {}
    for model in models:
        self_for_model = self_cases[self_cases["judge_model"] == model]
        self_n = len(self_for_model)
        self_wins = int((self_for_model["winner_model"] == model).sum())
        self_rate = self_wins / self_n if self_n else float("nan")

        model_in_pair = (neutral_cases["model_a"] == model) | (neutral_cases["model_b"] == model)
        baseline_for_model = neutral_cases[model_in_pair]
        baseline_n = len(baseline_for_model)
        baseline_wins = int((baseline_for_model["winner_model"] == model).sum())
        baseline_rate = baseline_wins / baseline_n if baseline_n else float("nan")

        chi2, p_value = _chi_square(self_wins, self_n, baseline_wins, baseline_n)

        # Position split: was the model's own argument shown first or second?
        # Applied to both the self cases AND the neutral baseline, so each
        # position row is a genuine self-vs-neutral comparison rather than
        # just the self rate on its own.
        self_first = self_for_model[self_for_model["first_model"] == model]
        self_second = self_for_model[self_for_model["second_model"] == model]

        n_first = len(self_first)
        wins_first = int((self_first["winner_model"] == model).sum())
        rate_first = wins_first / n_first if n_first else float("nan")

        n_second = len(self_second)
        wins_second = int((self_second["winner_model"] == model).sum())
        rate_second = wins_second / n_second if n_second else float("nan")

        baseline_first = baseline_for_model[baseline_for_model["first_model"] == model]
        baseline_second = baseline_for_model[baseline_for_model["second_model"] == model]

        baseline_n_first = len(baseline_first)
        baseline_wins_first = int((baseline_first["winner_model"] == model).sum())
        baseline_rate_first = (
            baseline_wins_first / baseline_n_first if baseline_n_first else float("nan")
        )

        baseline_n_second = len(baseline_second)
        baseline_wins_second = int((baseline_second["winner_model"] == model).sum())
        baseline_rate_second = (
            baseline_wins_second / baseline_n_second if baseline_n_second else float("nan")
        )

        chi2_first, p_first = _chi_square(
            wins_first, n_first, baseline_wins_first, baseline_n_first
        )
        chi2_second, p_second = _chi_square(
            wins_second, n_second, baseline_wins_second, baseline_n_second
        )

        per_model[model] = {
            "self_rate": self_rate,
            "self_n": self_n,
            "baseline_rate": baseline_rate,
            "baseline_n": baseline_n,
            "chi2": chi2,
            "p_value": p_value,
            "self_rate_first": rate_first,
            "n_first": n_first,
            "self_rate_second": rate_second,
            "n_second": n_second,
            "baseline_rate_first": baseline_rate_first,
            "baseline_n_first": baseline_n_first,
            "baseline_rate_second": baseline_rate_second,
            "baseline_n_second": baseline_n_second,
            "difference_first": rate_first - baseline_rate_first,
            "p_value_first": p_first,
            "difference_second": rate_second - baseline_rate_second,
            "p_value_second": p_second,
        }

    overall_self_rate = (
        (self_cases["winner_model"] == self_cases["judge_model"]).mean()
        if len(self_cases) else float("nan")
    )

    return {
        "overall_self_enhancement_rate": overall_self_rate,
        "n_self_judged_cases": len(self_cases),
        "by_model": per_model,
    }


def first_position_preference_rate(df):
    """
    Proportion of verdicts where the first-presented argument wins,
    tested against the null that an unbiased judge picks position one
    about half the time (binomial test, p=0.5).
    """
    total = len(df)
    first_wins = int((df["winner_model"] == df["first_model"]).sum())
    rate = first_wins / total if total else float("nan")
    test_result = binomtest(first_wins, total, 0.5) if total else None

    per_judge = {}
    for judge, group in df.groupby("judge_model"):
        n = len(group)
        wins = int((group["winner_model"] == group["first_model"]).sum())
        per_judge[judge] = {
            "rate": wins / n if n else float("nan"),
            "n": n,
            "p_value": binomtest(wins, n, 0.5).pvalue if n else float("nan"),
        }

    return {
        "first_position_preference_rate": rate,
        "n": total,
        "p_value": test_result.pvalue if test_result else float("nan"),
        "by_judge": per_judge,
    }


def _binom_rate(wins, n):
    rate = wins / n if n else float("nan")
    p_value = binomtest(wins, n, 0.5).pvalue if n else float("nan")
    return rate, p_value


def _verbosity_stats_for_subset(subset_df):
    """
    Core verbosity calculation for whatever subset of rows is passed in
    (pooled, or one judge's rows). Ties (equal word count) are excluded
    since there's no "longer" argument to evaluate for that comparison.
    """
    non_tied = subset_df[
        subset_df["first_arg_length_words"] != subset_df["second_arg_length_words"]
    ].copy()

    def longer_won(row):
        longer_model = (
            row["first_model"]
            if row["first_arg_length_words"] > row["second_arg_length_words"]
            else row["second_model"]
        )
        return row["winner_model"] == longer_model

    n = len(non_tied)
    if n:
        won_mask = non_tied.apply(longer_won, axis=1)
        longer_wins = int(won_mask.sum())
        rate, p_value = _binom_rate(longer_wins, n)

        longer_first_mask = non_tied["first_arg_length_words"] > non_tied["second_arg_length_words"]
        first_subset = won_mask[longer_first_mask]
        second_subset = won_mask[~longer_first_mask]

        rate_first, p_first = _binom_rate(int(first_subset.sum()), len(first_subset))
        rate_second, p_second = _binom_rate(int(second_subset.sum()), len(second_subset))
        n_first, n_second = len(first_subset), len(second_subset)
    else:
        rate = p_value = float("nan")
        rate_first = p_first = float("nan")
        rate_second = p_second = float("nan")
        n_first = n_second = 0

    return {
        "longer_arg_win_rate": rate,
        "n": n,
        "n_ties_excluded": len(subset_df) - n,
        "p_value": p_value,
        "rate_when_longer_first": rate_first,
        "n_when_longer_first": n_first,
        "p_when_longer_first": p_first,
        "rate_when_longer_second": rate_second,
        "n_when_longer_second": n_second,
        "p_when_longer_second": p_second,
    }


def verbosity_test(df):
    """
    One observation per comparison (not two): did the longer of the two
    arguments win? Tested against a null of 0.5 with a binomial test, and
    ALSO split by whether the longer argument was shown first or second.
    Computed pooled AND per judge, so a per-judge breakdown can show
    whether one specific judge is driving the pooled effect or whether
    it's consistent across all three.

    The position split is the actual finding here: if the longer argument
    only wins when it happens to be shown first, that's position bias,
    not a length preference -- the pooled rate can't distinguish the
    two, since a judge that just always picks position one will show a
    "verbosity effect" any time the longer argument happens to load
    first, purely by chance of the swap assignment.
    """
    pooled = _verbosity_stats_for_subset(df)

    by_judge = {
        judge: _verbosity_stats_for_subset(group)
        for judge, group in df.groupby("judge_model")
    }

    pooled["by_judge"] = by_judge
    return pooled


def confidence_stats(df_all):
    """Mean judge-reported confidence (1-5 scale), overall and per judge."""
    valid = df_all.dropna(subset=["confidence"])
    overall_mean = valid["confidence"].mean() if len(valid) else float("nan")
    by_judge = {
        judge: {"mean": group["confidence"].mean(), "n": len(group)}
        for judge, group in valid.groupby("judge_model")
    }
    return {"overall_mean": overall_mean, "n": len(valid), "by_judge": by_judge}


def response_time_stats(df_all):
    """Mean response time in seconds, overall and per judge (failed calls excluded, no timing)."""
    valid = df_all.dropna(subset=["response_time_sec"])
    overall_mean = valid["response_time_sec"].mean() if len(valid) else float("nan")
    by_judge = {
        judge: {"mean": group["response_time_sec"].mean(), "n": len(group)}
        for judge, group in valid.groupby("judge_model")
    }
    return {"overall_mean": overall_mean, "n": len(valid), "by_judge": by_judge}


def inter_judge_agreement(df):
    """
    Cohen's kappa between each pair of judges, restricted to items sharing
    the same (topic, model_a, model_b). Not also keyed on `swapped`: the
    judge script fixes the argument order per pair identically for all
    three judges, so swap state is no longer judge-specific, and
    `winner_model` is already the actual model name (not "Argument 1/2"),
    so order doesn't affect whether two judges' labels agree.
    """
    by_item = defaultdict(dict)
    for _, row in df.iterrows():
        key = (row["topic"], row["model_a"], row["model_b"])
        by_item[key][row["judge_model"]] = row["winner_model"]

    judges = sorted(df["judge_model"].unique())
    kappas = {}
    for judge_x, judge_y in itertools.combinations(judges, 2):
        labels_x, labels_y = [], []
        for verdicts in by_item.values():
            if judge_x in verdicts and judge_y in verdicts:
                labels_x.append(verdicts[judge_x])
                labels_y.append(verdicts[judge_y])
        if len(labels_x) >= 2:
            kappas[f"{judge_x} vs {judge_y}"] = {
                "kappa": cohen_kappa_score(labels_x, labels_y),
                "n_shared_items": len(labels_x),
            }
        else:
            kappas[f"{judge_x} vs {judge_y}"] = {"kappa": float("nan"), "n_shared_items": len(labels_x)}

    return kappas


def main():
    df_all = load_all()

    argument_texts = load_argument_texts()
    tied_pairs = find_content_tied_pairs(df_all, argument_texts)
    print(f"Content-tied pairs (identical argument text on both sides): "
          f"{len(tied_pairs)} found.")
    if tied_pairs:
        print(f"  Excluding {len(tied_pairs)} tied pair(s) from headline metrics: "
              f"{sorted(tied_pairs)[:5]}{'...' if len(tied_pairs) > 5 else ''}")
        df_all = drop_content_tied(df_all, tied_pairs)

    df = load_decisive(df_all)
    print(f"Loaded {len(df_all)} total verdicts; {len(df)} decisive (winner picked).\n")

    print("=== Outcome Rates ===")
    rates = outcome_rates(df_all)
    for outcome, result in rates.items():
        print(f"  {outcome}: {result['rate']:.3f} (n={result['count']})")

    print("\n=== Self-Enhancement Rate (per model, split by position) ===")
    se = self_enhancement_rate(df)
    print(f"Overall pooled self-enhancement rate: {se['overall_self_enhancement_rate']:.3f} "
          f"(n={se['n_self_judged_cases']}) -- see per-model breakdown below; "
          f"the pooled number can mask models cancelling each other out.")
    for model, result in se["by_model"].items():
        print(f"  {model}: self={result['self_rate']:.3f} (n={result['self_n']}), "
              f"baseline={result['baseline_rate']:.3f} (n={result['baseline_n']}), "
              f"chi2 p={result['p_value']:.4f}")
        print(f"    when self shown first:  self={result['self_rate_first']:.3f} "
              f"(n={result['n_first']}), baseline={result['baseline_rate_first']:.3f} "
              f"(n={result['baseline_n_first']}), diff={result['difference_first']:+.3f}, "
              f"p={result['p_value_first']:.4f}")
        print(f"    when self shown second: self={result['self_rate_second']:.3f} "
              f"(n={result['n_second']}), baseline={result['baseline_rate_second']:.3f} "
              f"(n={result['baseline_n_second']}), diff={result['difference_second']:+.3f}, "
              f"p={result['p_value_second']:.4f}")

    print("\n=== First-Position Preference Rate ===")
    fp = first_position_preference_rate(df)
    print(f"Overall: {fp['first_position_preference_rate']:.3f} "
          f"(n={fp['n']}, p={fp['p_value']:.4f})")
    print("By judge:")
    for judge, result in fp["by_judge"].items():
        print(f"  {judge}: rate={result['rate']:.3f} (n={result['n']}, p={result['p_value']:.4f})")

    print("\n=== Verbosity Test (split by position) ===")
    vt = verbosity_test(df)
    print(f"Pooled longer-argument win rate: {vt['longer_arg_win_rate']:.3f} "
          f"(n={vt['n']}, {vt['n_ties_excluded']} ties excluded, p={vt['p_value']:.4f})")
    print(f"  when longer shown first:  {vt['rate_when_longer_first']:.3f} "
          f"(n={vt['n_when_longer_first']}, p={vt['p_when_longer_first']:.4f})")
    print(f"  when longer shown second: {vt['rate_when_longer_second']:.3f} "
          f"(n={vt['n_when_longer_second']}, p={vt['p_when_longer_second']:.4f})")

    def _report_flip_status(rf, rs, label, indent="  "):
        if rf == rf and rs == rs:  # neither is NaN
            both_above_half = rf > 0.5 and rs > 0.5
            both_below_half = rf < 0.5 and rs < 0.5
            if both_above_half or both_below_half:
                print(f"{indent}-> {label}: consistent across both positions "
                      f"(first={rf:.3f}, second={rs:.3f}) -- supports a genuine "
                      f"length effect, though check the p-values above for "
                      f"significance at this n.")
            else:
                print(f"{indent}-> {label}: FLIPS between positions "
                      f"(first={rf:.3f}, second={rs:.3f}) -- likely position "
                      f"bias, not a real verbosity preference.")
        else:
            print(f"{indent}-> {label}: not enough data in one or both "
                  f"position splits to compare.")

    # Headline call: does the verbosity effect hold up in BOTH positions,
    # or does it only appear in one -- which would mean it's actually
    # position bias, not a genuine preference for longer arguments.
    _report_flip_status(vt["rate_when_longer_first"], vt["rate_when_longer_second"], "Pooled")

    print("\n  By judge (does one judge drive the pooled effect, or is it consistent?):")
    for judge, result in vt["by_judge"].items():
        print(f"    {judge}: pooled={result['longer_arg_win_rate']:.3f} "
              f"(n={result['n']}, p={result['p_value']:.4f})")
        print(f"      when longer first:  {result['rate_when_longer_first']:.3f} "
              f"(n={result['n_when_longer_first']}, p={result['p_when_longer_first']:.4f})")
        print(f"      when longer second: {result['rate_when_longer_second']:.3f} "
              f"(n={result['n_when_longer_second']}, p={result['p_when_longer_second']:.4f})")
        _report_flip_status(
            result["rate_when_longer_first"], result["rate_when_longer_second"],
            judge, indent="      "
        )

    print("\n=== Confidence ===")
    cs = confidence_stats(df_all)
    print(f"Overall mean confidence: {cs['overall_mean']:.2f} (n={cs['n']})")
    for judge, result in cs["by_judge"].items():
        print(f"  {judge}: mean={result['mean']:.2f} (n={result['n']})")

    print("\n=== Response Time ===")
    rt = response_time_stats(df_all)
    print(f"Overall mean response time: {rt['overall_mean']:.3f}s (n={rt['n']})")
    for judge, result in rt["by_judge"].items():
        print(f"  {judge}: mean={result['mean']:.3f}s (n={result['n']})")

    print("\n=== Inter-Judge Agreement (Cohen's kappa) ===")
    kappas = inter_judge_agreement(df)
    for pair, result in kappas.items():
        print(f"  {pair}: kappa={result['kappa']:.3f} (n={result['n_shared_items']})")

    print("\n=== Ground Truth Accuracy ===")
    print("N/A -- ArgKP arguments have no correct answer; not applicable to this dataset.")


if __name__ == "__main__":
    main()