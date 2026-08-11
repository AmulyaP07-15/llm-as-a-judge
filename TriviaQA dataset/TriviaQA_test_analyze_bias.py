"""
Unit tests for the TriviaQA objective analysis metrics.

Each test builds synthetic verdict data with properties known in advance, runs
one metric against it, and asserts that the number coming back is the number
that should come back. The synthetic builders live at module level and are
shared across tests.
"""
import itertools

import pandas as pd

import TriviaQA_analyze_bias as an

MODELS = ["allam", "llama", "qwen"]


def assert_close(got, want, tol=1e-9):
    """Assert two numbers match, treating NaN as equal to NaN."""
    ok = (pd.isna(got) and pd.isna(want)) or abs(got - want) < tol
    assert ok, f"got {got}, want {want}"


def make_rows(n_questions, choice_fn, len_fn=None, answer_fn=None):
    """
    One row per question, pair, judge. All three judges see the same pair in
    the same order, which mirrors how the real pipeline works.
    """
    rows = []
    for q in range(n_questions):
        for a, b in itertools.combinations(MODELS, 2):
            first, second = (a, b) if q % 2 == 0 else (b, a)
            la, lb = len_fn(q) if len_fn else (5, 5)
            for judge in MODELS:
                ch = choice_fn(q, first, second, judge)
                winner = first if ch == "1" else second if ch == "2" else "NEITHER"
                fa, sa = answer_fn(q, first, second) if answer_fn else ("x", "y")
                rows.append({
                    "judge_model": judge,
                    "question": f"q{q}",
                    "correct_answer": "target",
                    "model_a": a,
                    "model_b": b,
                    "swapped": q % 2 == 1,
                    "first_model": first,
                    "second_model": second,
                    "first_answer": fa,
                    "second_answer": sa,
                    "first_length_words": la,
                    "second_length_words": lb,
                    "raw_verdict": "",
                    "choice": ch,
                    "winner_model": winner,
                    "confidence": 4,
                    "correct_match": None,
                    "parse_method": "labeled",
                    "response_time": 0.5,
                })
    return pd.DataFrame(rows)


def llama_favours_itself(q, first, second, judge):
    """llama always picks itself. Everyone else takes the first option."""
    if judge == "llama":
        return "1" if first == "llama" else "2" if second == "llama" else "1"
    return "1"


def first_only(q, first, second, judge):
    """llama favours itself only when shown first."""
    if judge == "llama" and first == "llama":
        return "1"
    return "1" if q % 2 == 0 else "2"


def picks_longer(q, f, s, j):
    """Whichever slot holds the longer answer is chosen."""
    return "1" if q % 2 == 0 else "2"


def discrimination_answers(q, f, s):
    """q0 has one right answer, q1 both wrong, q2 both right."""
    if q == 0:
        return ("Port Moresby", "Sydney")
    if q == 1:
        return ("Sydney", "Cairo")
    return ("Port Moresby", "port moresby city")


# ---------------------------------------------------------------------------
# 1. First position preference
# ---------------------------------------------------------------------------
def test_position_all_first_gives_rate_one():
    # every judge always takes the first option, so the rate must be exactly 1.0
    df = make_rows(10, lambda q, f, s, j: "1")
    res = an.first_position_preference(an.decisive(df))
    assert_close(res[res["judge"] == "ALL"]["rate"].iloc[0], 1.0)


def test_position_alternating_gives_half():
    # alternating by question gives an exact half, and a rate of 0.5 must not
    # be flagged as significant
    df = make_rows(10, lambda q, f, s, j: "1" if q % 2 == 0 else "2")
    res = an.first_position_preference(an.decisive(df))
    row = res[res["judge"] == "ALL"]
    assert_close(row["rate"].iloc[0], 0.5)
    assert row["p_value"].iloc[0] > 0.05


# ---------------------------------------------------------------------------
# 2. Self enhancement
# ---------------------------------------------------------------------------
def test_self_enhancement_llama_self_rate():
    df = make_rows(20, llama_favours_itself)
    res = an.self_enhancement(an.decisive(df))
    llama = res[res["model"] == "llama"].iloc[0]
    assert_close(llama["self_rate"], 1.0)


def test_self_enhancement_llama_neutral_rate():
    # neutral judge is allam or qwen, who always take the first option. order
    # alternates by question, so llama sits first half the time.
    df = make_rows(20, llama_favours_itself)
    res = an.self_enhancement(an.decisive(df))
    llama = res[res["model"] == "llama"].iloc[0]
    assert_close(llama["neutral_rate"], 0.5)


def test_self_enhancement_llama_difference():
    df = make_rows(20, llama_favours_itself)
    res = an.self_enhancement(an.decisive(df))
    llama = res[res["model"] == "llama"].iloc[0]
    assert_close(llama["difference"], 0.5)


def test_self_enhancement_neutral_n_excludes_opponent():
    # the baseline must exclude the opponent's own judgments. pairs containing
    # llama are llama vs allam and llama vs qwen, 20 questions each, so the
    # third model judges 40 of them. if the opponent leaked in it would be 80.
    df = make_rows(20, llama_favours_itself)
    res = an.self_enhancement(an.decisive(df))
    llama = res[res["model"] == "llama"].iloc[0]
    assert_close(llama["neutral_n"], 40)


# ---------------------------------------------------------------------------
# 3. Verbosity
# ---------------------------------------------------------------------------
def test_verbosity_longer_always_wins():
    # first answer longer on even questions, shorter on odd, always pick longer
    df = make_rows(20,
                   lambda q, f, s, j: "1" if q % 2 == 0 else "2",
                   len_fn=lambda q: (10, 3) if q % 2 == 0 else (3, 10))
    res = an.verbosity_preference(an.decisive(df))
    assert_close(res[res["judge"] == "ALL"]["longer_wins_rate"].iloc[0], 1.0)


def test_verbosity_equal_lengths_dropped():
    # same lengths carry no signal and should be dropped entirely
    df = make_rows(20, lambda q, f, s, j: "1", len_fn=lambda q: (5, 5))
    res = an.verbosity_preference(an.decisive(df))
    assert_close(res[res["judge"] == "ALL"]["n"].iloc[0], 0)


# ---------------------------------------------------------------------------
# 4. Ground truth matching
# ---------------------------------------------------------------------------
PORT_MORESBY = {"port moresby", "moresby", "pg-ncd"}
# the muppet problem. a short answer must not match a longer alias, otherwise
# "rat" scores correct against "yolanda rat"
MUPPETS = {"yolanda rat", "constantine frog", "robin"}


def test_ground_truth_exact():
    assert_close(float(an.is_correct("Port Moresby", PORT_MORESBY)), 1.0)


def test_ground_truth_padded():
    assert_close(float(an.is_correct("The capital is Port Moresby.", PORT_MORESBY)), 1.0)


def test_ground_truth_wrong():
    assert_close(float(an.is_correct("Sydney", PORT_MORESBY)), 0.0)


def test_ground_truth_short_answer_not_matched_to_longer_alias():
    assert_close(float(an.is_correct("Rat", MUPPETS)), 0.0)


def test_ground_truth_exact_still_allowed_at_that_length():
    assert_close(float(an.is_correct("Robin", MUPPETS)), 1.0)


def test_ground_truth_short_alias_exact():
    # aliases under the length floor only match exactly
    assert_close(float(an.is_correct("PG", {"pg"})), 1.0)


def test_ground_truth_short_alias_not_substring():
    assert_close(float(an.is_correct("PG tips tea", {"pg"})), 0.0)


# ---------------------------------------------------------------------------
# 5. Inter judge agreement
# ---------------------------------------------------------------------------
def test_agreement_perfect_kappa():
    # every judge gives the same verdict, so kappa is 1.0
    df = make_rows(20, lambda q, f, s, j: "1" if q % 3 else "2")
    res = an.inter_judge_agreement(df)
    assert_close(res["kappa"].iloc[0], 1.0)


def test_agreement_shared_items():
    # 20 questions times 3 pairs is 60 shared items for any judge pair. if
    # presentation order were in the item key this would come out near 30.
    df = make_rows(20, lambda q, f, s, j: "1" if q % 3 else "2")
    res = an.inter_judge_agreement(df)
    assert_close(res["n_shared"].iloc[0], 60)


# ---------------------------------------------------------------------------
# 6. Neither and unparsed are kept separate
# ---------------------------------------------------------------------------
def test_decisive_excludes_unparsed_and_abstain():
    df = make_rows(10, lambda q, f, s, j: "NEITHER" if q < 5 else "1")
    df.loc[df.index[:6], "winner_model"] = None  # simulate parse failures
    decisive_rows = an.decisive(df)
    assert not decisive_rows["winner_model"].isna().any()
    assert not (decisive_rows["winner_model"] == "NEITHER").any()


# ---------------------------------------------------------------------------
# 7. Punctuation normalisation
# ---------------------------------------------------------------------------
SEESAW = {an.normalise(a) for a in ["see saw", "seesaw", "teeter totter"]}
APOSTROPHE = {an.normalise(a) for a in ["mcdonald's", "mackey d's"]}


def test_normalise_hyphen_vs_space():
    assert_close(float(an.is_correct("See-saw", SEESAW)), 1.0)


def test_normalise_trailing_period():
    assert_close(float(an.is_correct("Seesaw.", SEESAW)), 1.0)


def test_normalise_still_rejects_wrong():
    assert_close(float(an.is_correct("Swing", SEESAW)), 0.0)


def test_normalise_apostrophe():
    assert_close(float(an.is_correct("McDonalds", APOSTROPHE)), 1.0)


def test_normalise_keeps_direction_rule():
    # normalisation must not make short answers match longer aliases
    assert_close(float(an.is_correct("Rat", {an.normalise("yolanda rat")})), 0.0)


# ---------------------------------------------------------------------------
# 8. Tie detection
# ---------------------------------------------------------------------------
def _all_tie_frame():
    # both answers identical after normalisation, so every row is a tie
    df = make_rows(10, lambda q, f, s, j: "1",
                   answer_fn=lambda q, f, s: ("Port Moresby", "port moresby."))
    return an.add_correctness(df, {f"q{i}": {"port moresby"} for i in range(10)})


def test_ties_all_detected():
    df = _all_tie_frame()
    assert_close(float(df["is_tie"].all()), 1.0)


def test_ties_rate_one():
    df = _all_tie_frame()
    res = an.tie_summary(an.decisive(df))
    assert_close(res["tie_rate"].iloc[0], 1.0)


def test_ties_first_pos_rate_on_ties():
    df = _all_tie_frame()
    res = an.tie_summary(an.decisive(df))
    assert_close(res["first_pos_rate_on_ties"].iloc[0], 1.0)


def test_ties_no_false_ties():
    # distinct answers, nothing should be flagged as a tie
    df = make_rows(10, lambda q, f, s, j: "1",
                   answer_fn=lambda q, f, s: (f"{f} answer", f"{s} answer"))
    df = an.add_correctness(df, {f"q{i}": {"whatever"} for i in range(10)})
    assert_close(float(df["is_tie"].any()), 0.0)


# ---------------------------------------------------------------------------
# 9. Verbosity controlled for position
# ---------------------------------------------------------------------------
def _pure_position_verbosity_frame():
    # judge always takes the first option regardless of length. that is pure
    # position bias.
    df = make_rows(20, lambda q, f, s, j: "1",
                   len_fn=lambda q: (10, 3) if q % 2 == 0 else (3, 10),
                   answer_fn=lambda q, f, s: (f"{f} ans", f"{s} ans"))
    return an.add_correctness(df, {f"q{i}": {"x"} for i in range(20)})


def test_verbosity_pure_position_split():
    # the split must show 1.0 when the longer answer is first and 0.0 when it
    # is second, rather than a misleading pooled figure
    df = _pure_position_verbosity_frame()
    res = an.verbosity_by_position(an.decisive(df[~df["is_tie"]]))
    first = res[(res["judge"] == "llama") & (res["longer_shown"] == "first")]
    second = res[(res["judge"] == "llama") & (res["longer_shown"] == "second")]
    assert_close(first["longer_wins_rate"].iloc[0], 1.0)
    assert_close(second["longer_wins_rate"].iloc[0], 0.0)


def test_verbosity_pooled_hides_confound_as_half():
    df = _pure_position_verbosity_frame()
    pooled = an.verbosity_preference(an.decisive(df[~df["is_tie"]]))
    rate = pooled[pooled["judge"] == "llama"]["longer_wins_rate"].iloc[0]
    assert_close(rate, 0.5)


def _real_preference_verbosity_frame():
    # genuine length preference, longer wins wherever it sits
    df = make_rows(20, picks_longer,
                   len_fn=lambda q: (10, 3) if q % 2 == 0 else (3, 10),
                   answer_fn=lambda q, f, s: (f"{f} ans", f"{s} ans"))
    return an.add_correctness(df, {f"q{i}": {"x"} for i in range(20)})


def test_verbosity_real_preference_first():
    df = _real_preference_verbosity_frame()
    res = an.verbosity_by_position(an.decisive(df[~df["is_tie"]]))
    both = res[res["judge"] == "llama"]["longer_wins_rate"].tolist()
    assert_close(both[0], 1.0)


def test_verbosity_real_preference_second():
    df = _real_preference_verbosity_frame()
    res = an.verbosity_by_position(an.decisive(df[~df["is_tie"]]))
    both = res[res["judge"] == "llama"]["longer_wins_rate"].tolist()
    assert_close(both[1], 1.0)


# ---------------------------------------------------------------------------
# 10. Discrimination accuracy
# ---------------------------------------------------------------------------
def _discrimination_frame():
    df = make_rows(3, lambda q, f, s, j: "1", answer_fn=discrimination_answers)
    return an.add_correctness(df, {f"q{i}": {"port moresby"} for i in range(3)})


def test_discrimination_one_right_counted():
    df = _discrimination_frame()
    cases = an.case_breakdown(an.decisive(df))
    assert_close(float(cases[cases["case"] == "one right"]["n"].iloc[0]), 9.0)


def test_discrimination_both_wrong_counted():
    df = _discrimination_frame()
    cases = an.case_breakdown(an.decisive(df))
    assert_close(float(cases[cases["case"] == "both wrong"]["n"].iloc[0]), 9.0)


def test_discrimination_accuracy_on_one_right():
    # a judge always taking the first option scores 1.0 on the one right case,
    # since the correct answer is always placed first here
    df = _discrimination_frame()
    res = an.discrimination_accuracy(an.decisive(df))
    assert_close(res[res["judge"] == "ALL"]["accuracy"].iloc[0], 1.0)


def test_discrimination_n_is_one_right_subset_only():
    df = _discrimination_frame()
    res = an.discrimination_accuracy(an.decisive(df))
    assert_close(float(res[res["judge"] == "ALL"]["n"].iloc[0]), 9.0)


# ---------------------------------------------------------------------------
# 11. Verbosity spearman
# ---------------------------------------------------------------------------
def test_spearman_positive_when_longer_wins():
    # longer answer always wins, so a bigger positive gap tracks with the first
    # answer winning and the correlation must be positive. under pure position,
    # where the choice never tracks length, correlation must be near zero.
    df = make_rows(20, lambda q, f, s, j: "1" if q % 2 == 0 else "2",
                   len_fn=lambda q: (12, 3) if q % 2 == 0 else (3, 12),
                   answer_fn=lambda q, f, s: (f"{f} ans", f"{s} ans"))
    df = an.add_correctness(df, {f"q{i}": {"x"} for i in range(20)})
    res = an.verbosity_correlation(an.decisive(df[~df["is_tie"]]))
    assert_close(res[res["judge"] == "ALL"]["spearman"].iloc[0], 1.0)

    df = make_rows(20, lambda q, f, s, j: "1",
                   len_fn=lambda q: (12, 3) if q % 2 == 0 else (3, 12),
                   answer_fn=lambda q, f, s: (f"{f} ans", f"{s} ans"))
    df = an.add_correctness(df, {f"q{i}": {"x"} for i in range(20)})
    res = an.verbosity_correlation(an.decisive(df[~df["is_tie"]]))
    rho = res[res["judge"] == "ALL"]["spearman"].iloc[0]
    assert pd.isna(rho) or abs(rho) < 0.01


def test_spearman_n_matches_comparisons_not_answers():
    # one row per comparison, not two
    df = make_rows(10, lambda q, f, s, j: "1",
                   len_fn=lambda q: (8, 3),
                   answer_fn=lambda q, f, s: (f"{f} ans", f"{s} ans"))
    df = an.add_correctness(df, {f"q{i}": {"x"} for i in range(10)})
    d = an.decisive(df[~df["is_tie"]])
    res = an.verbosity_correlation(d)
    assert_close(float(res[res["judge"] == "ALL"]["n"].iloc[0]), float(len(d)))
