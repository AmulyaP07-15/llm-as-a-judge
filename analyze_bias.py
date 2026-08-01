import itertools
from collections import defaultdict

import pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.metrics import cohen_kappa_score

VERDICTS_FILE = "judge_verdicts.csv"

NON_DECISIVE = {"NEITHER", "UNCLEAR", "API_ERROR"}


def load_all(path=VERDICTS_FILE):
    return pd.read_csv(path)


def load_decisive(df):
    """Rows with an actual winning model -- excludes NEITHER/UNCLEAR/API_ERROR."""
    return df[~df["winner_model"].isin(NON_DECISIVE)].copy()


def outcome_rates(df):
    """Proportion of all verdicts that were NEITHER, UNCLEAR, or API_ERROR."""
    total = len(df)
    return {
        outcome: (df["winner_model"] == outcome).mean() if total else float("nan")
        for outcome in NON_DECISIVE
    }


def self_enhancement_rate(df):
   
    is_self_case = (df["judge_model"] == df["model_a"]) | (df["judge_model"] == df["model_b"])
    self_cases = df[is_self_case]
    neutral_cases = df[~is_self_case]

    self_rate = (
        (self_cases["winner_model"] == self_cases["judge_model"]).mean()
        if len(self_cases) else float("nan")
    )

    baseline_rates = {}
    for model in pd.concat([df["model_a"], df["model_b"]]).unique():
        model_in_pair = (neutral_cases["model_a"] == model) | (neutral_cases["model_b"] == model)
        relevant = neutral_cases[model_in_pair]
        if len(relevant):
            baseline_rates[model] = (relevant["winner_model"] == model).mean()

    return {
        "self_enhancement_rate": self_rate,
        "n_self_judged_cases": len(self_cases),
        "neutral_win_rate_by_model": baseline_rates,
    }


def first_position_preference_rate(df):
   
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


def verbosity_correlation(df):
  
    lengths = []
    wins = []
    for _, row in df.iterrows():
        lengths.append(row["first_arg_length_words"])
        wins.append(1 if row["winner_model"] == row["first_model"] else 0)
        lengths.append(row["second_arg_length_words"])
        wins.append(1 if row["winner_model"] == row["second_model"] else 0)

    corr, p_value = spearmanr(lengths, wins)
    return {"verbosity_spearman_corr": corr, "p_value": p_value, "n_observations": len(lengths)}


def inter_judge_agreement(df):
   
    by_item = defaultdict(dict)
    for _, row in df.iterrows():
        key = (row["topic"], row["model_a"], row["model_b"], row["swapped"])
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
    df = load_decisive(df_all)
    print(f"Loaded {len(df_all)} total verdicts; {len(df)} decisive (winner picked).\n")

    print("=== Outcome Rates ===")
    rates = outcome_rates(df_all)
    for outcome, rate in rates.items():
        print(f"  {outcome}: {rate:.3f}")

    print("\n=== Self-Enhancement Rate ===")
    se = self_enhancement_rate(df)
    print(f"Self-enhancement rate: {se['self_enhancement_rate']:.3f} "
          f"(n={se['n_self_judged_cases']})")
    print("Neutral (baseline) win rate by model:")
    for model, rate in se["neutral_win_rate_by_model"].items():
        print(f"  {model}: {rate:.3f}")

    print("\n=== First-Position Preference Rate ===")
    fp = first_position_preference_rate(df)
    print(f"Overall: {fp['first_position_preference_rate']:.3f} "
          f"(n={fp['n']}, p={fp['p_value']:.4f})")
    print("By judge:")
    for judge, result in fp["by_judge"].items():
        print(f"  {judge}: rate={result['rate']:.3f} (n={result['n']}, p={result['p_value']:.4f})")

    print("\n=== Verbosity Correlation ===")
    vc = verbosity_correlation(df)
    print(f"Spearman correlation (length vs. win): {vc['verbosity_spearman_corr']:.3f} "
          f"(p={vc['p_value']:.4f}, n={vc['n_observations']})")

    print("\n=== Inter-Judge Agreement (Cohen's kappa) ===")
    kappas = inter_judge_agreement(df)
    for pair, result in kappas.items():
        print(f"  {pair}: kappa={result['kappa']:.3f} (n={result['n_shared_items']})")


if __name__ == "__main__":
    main()