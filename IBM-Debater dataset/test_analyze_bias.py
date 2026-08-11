import math

import pandas as pd
import pytest

import analyze_bias as ab


def _row(judge_model, model_a, model_b, winner_model, first_model=None, second_model=None,
         first_len=None, second_len=None, confidence=None, response_time=None, topic="t"):
    first_model = first_model or model_a
    second_model = second_model or model_b
    return {
        "judge_model": judge_model,
        "topic": topic,
        "model_a": model_a,
        "model_b": model_b,
        "first_model": first_model,
        "second_model": second_model,
        "first_arg_length_words": first_len,
        "second_arg_length_words": second_len,
        "winner_model": winner_model,
        "confidence": confidence,
        "response_time_sec": response_time,
    }


class TestLoadArgumentTexts:
    def test_builds_topic_model_lookup_stripped(self, tmp_path):
        path = tmp_path / "arguments.csv"
        pd.DataFrame([
            {"model": "llama", "topic": "t1", "argument": "  hello  "},
            {"model": "qwen", "topic": "t1", "argument": "world"},
        ]).to_csv(path, index=False)

        result = ab.load_argument_texts(str(path))

        assert result == {("t1", "llama"): "hello", ("t1", "qwen"): "world"}


class TestFindContentTiedPairs:
    def test_detects_identical_arguments(self):
        df = pd.DataFrame([_row("llama", "a", "b", "a")])
        texts = {("t", "a"): "same text", ("t", "b"): "same text"}

        tied = ab.find_content_tied_pairs(df, texts)

        assert tied == {("t", "a", "b")}

    def test_different_arguments_are_not_tied(self):
        df = pd.DataFrame([_row("llama", "a", "b", "a")])
        texts = {("t", "a"): "one thing", ("t", "b"): "another thing"}

        assert ab.find_content_tied_pairs(df, texts) == set()


class TestDropContentTied:
    def test_removes_only_the_tied_pair(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a", topic="t1"),
            _row("llama", "a", "c", "a", topic="t1"),
        ])

        result = ab.drop_content_tied(df, {("t1", "a", "b")})

        assert len(result) == 1
        assert result.iloc[0]["model_b"] == "c"

    def test_no_tied_pairs_returns_df_unchanged(self):
        df = pd.DataFrame([_row("llama", "a", "b", "a")])
        result = ab.drop_content_tied(df, set())
        assert len(result) == len(df)


class TestLoadDecisive:
    def test_excludes_non_decisive_outcomes(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a"),
            _row("llama", "a", "b", "NEITHER"),
            _row("llama", "a", "b", "UNCLEAR"),
            _row("llama", "a", "b", "API_ERROR"),
            _row("llama", "a", "b", "ECHOED_PROMPT"),
        ])

        result = ab.load_decisive(df)

        assert len(result) == 1
        assert result.iloc[0]["winner_model"] == "a"


class TestOutcomeRates:
    def test_counts_and_rates_for_each_non_decisive_outcome(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a"),
            _row("llama", "a", "b", "NEITHER"),
            _row("llama", "a", "b", "NEITHER"),
            _row("llama", "a", "b", "UNCLEAR"),
        ])

        rates = ab.outcome_rates(df)

        assert rates["NEITHER"] == {"rate": pytest.approx(0.5), "count": 2}
        assert rates["UNCLEAR"] == {"rate": pytest.approx(0.25), "count": 1}
        assert rates["API_ERROR"] == {"rate": 0.0, "count": 0}
        assert rates["ECHOED_PROMPT"] == {"rate": 0.0, "count": 0}


class TestSelfEnhancementRate:
    def test_self_rate_vs_baseline_and_position_split(self):
        df = pd.DataFrame([
            # llama judges itself, its own argument shown first, and wins
            _row("llama", "llama", "qwen", "llama", first_model="llama", second_model="qwen"),
            # llama judges itself, its own argument shown second, and loses
            _row("llama", "qwen", "llama", "qwen", first_model="qwen", second_model="llama"),
            # neutral case: allam judges llama vs qwen
            _row("allam", "llama", "qwen", "llama", first_model="llama", second_model="qwen"),
        ])

        result = ab.self_enhancement_rate(df)

        llama_stats = result["by_model"]["llama"]
        assert llama_stats["self_n"] == 2
        assert llama_stats["self_rate"] == pytest.approx(0.5)
        assert llama_stats["baseline_n"] == 1
        assert llama_stats["baseline_rate"] == pytest.approx(1.0)
        assert llama_stats["n_first"] == 1
        assert llama_stats["self_rate_first"] == pytest.approx(1.0)
        assert llama_stats["n_second"] == 1
        assert llama_stats["self_rate_second"] == pytest.approx(0.0)

    def test_baseline_split_by_position(self):
        df = pd.DataFrame([
            # self cases: llama judges itself, own argument first (win) then second (loss)
            _row("llama", "llama", "qwen", "llama", first_model="llama", second_model="qwen"),
            _row("llama", "qwen", "llama", "qwen", first_model="qwen", second_model="llama"),
            # neutral cases, llama's argument shown first: allam judges llama vs qwen, wins both times
            _row("allam", "llama", "qwen", "llama", first_model="llama", second_model="qwen"),
            _row("allam", "llama", "qwen", "llama", first_model="llama", second_model="qwen"),
            # neutral case, llama's argument shown second: allam judges qwen vs llama, llama loses
            _row("allam", "qwen", "llama", "qwen", first_model="qwen", second_model="llama"),
        ])

        result = ab.self_enhancement_rate(df)

        llama_stats = result["by_model"]["llama"]
        assert llama_stats["baseline_n_first"] == 2
        assert llama_stats["baseline_rate_first"] == pytest.approx(1.0)
        assert llama_stats["baseline_n_second"] == 1
        assert llama_stats["baseline_rate_second"] == pytest.approx(0.0)
        # self=1.0, baseline=1.0 in position one -> no gap once position is held fixed
        assert llama_stats["difference_first"] == pytest.approx(0.0)
        # self=0.0, baseline=0.0 in position two -> also no gap
        assert llama_stats["difference_second"] == pytest.approx(0.0)

    def test_no_self_judged_cases_gives_nan_overall_rate(self):
        df = pd.DataFrame([_row("allam", "llama", "qwen", "llama")])

        result = ab.self_enhancement_rate(df)

        assert result["n_self_judged_cases"] == 0
        assert math.isnan(result["overall_self_enhancement_rate"])


class TestFirstPositionPreferenceRate:
    def test_rate_reflects_how_often_first_model_wins(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a", first_model="a", second_model="b"),
            _row("llama", "a", "b", "a", first_model="a", second_model="b"),
            _row("llama", "a", "b", "b", first_model="a", second_model="b"),
            _row("qwen", "a", "b", "b", first_model="a", second_model="b"),
        ])

        result = ab.first_position_preference_rate(df)

        assert result["n"] == 4
        assert result["first_position_preference_rate"] == pytest.approx(0.5)
        assert result["by_judge"]["llama"]["rate"] == pytest.approx(2 / 3)
        assert result["by_judge"]["qwen"]["rate"] == pytest.approx(0.0)


class TestVerbosityTest:
    def test_pooled_and_position_split(self):
        df = pd.DataFrame([
            # longer (first, 10 words) wins
            _row("llama", "a", "b", "a", first_model="a", second_model="b",
                 first_len=10, second_len=3),
            # longer (second, 8 words) loses -- first (2 words) wins instead
            _row("llama", "a", "b", "a", first_model="a", second_model="b",
                 first_len=2, second_len=8),
            # tie in length -- excluded
            _row("llama", "a", "b", "a", first_model="a", second_model="b",
                 first_len=5, second_len=5),
        ])

        result = ab.verbosity_test(df)

        assert result["n"] == 2
        assert result["n_ties_excluded"] == 1
        assert result["longer_arg_win_rate"] == pytest.approx(0.5)
        assert result["rate_when_longer_first"] == pytest.approx(1.0)
        assert result["rate_when_longer_second"] == pytest.approx(0.0)

    def test_no_usable_rows_returns_nan(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a", first_model="a", second_model="b",
                 first_len=5, second_len=5),
        ])

        result = ab.verbosity_test(df)

        assert result["n"] == 0
        assert math.isnan(result["longer_arg_win_rate"])


class TestConfidenceStats:
    def test_ignores_missing_confidence(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a", confidence=4),
            _row("llama", "a", "b", "a", confidence=None),
            _row("qwen", "a", "b", "a", confidence=2),
        ])

        result = ab.confidence_stats(df)

        assert result["n"] == 2
        assert result["overall_mean"] == pytest.approx(3.0)
        assert result["by_judge"]["llama"]["mean"] == pytest.approx(4.0)
        assert result["by_judge"]["qwen"]["mean"] == pytest.approx(2.0)


class TestResponseTimeStats:
    def test_ignores_missing_response_time(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a", response_time=1.0),
            _row("llama", "a", "b", "a", response_time=None),
            _row("qwen", "a", "b", "a", response_time=3.0),
        ])

        result = ab.response_time_stats(df)

        assert result["n"] == 2
        assert result["overall_mean"] == pytest.approx(2.0)


class TestInterJudgeAgreement:
    def test_perfect_agreement_gives_kappa_one(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a", topic="t1"),
            _row("qwen", "a", "b", "a", topic="t1"),
            _row("llama", "a", "c", "c", topic="t2"),
            _row("qwen", "a", "c", "c", topic="t2"),
            _row("llama", "a", "d", "a", topic="t3"),
            _row("qwen", "a", "d", "d", topic="t3"),
        ])

        result = ab.inter_judge_agreement(df)

        assert result["llama vs qwen"]["n_shared_items"] == 3
        assert result["llama vs qwen"]["kappa"] < 1.0  # disagree on t3

    def test_no_shared_items_gives_nan_kappa(self):
        df = pd.DataFrame([
            _row("llama", "a", "b", "a", topic="t1"),
            _row("qwen", "a", "c", "a", topic="t2"),
        ])

        result = ab.inter_judge_agreement(df)

        assert math.isnan(result["llama vs qwen"]["kappa"])
        assert result["llama vs qwen"]["n_shared_items"] == 0


class TestChiSquare:
    def test_degenerate_all_wins_returns_nan(self):
        chi2, p = ab._chi_square(5, 5, 5, 5)
        assert math.isnan(chi2)
        assert math.isnan(p)

    def test_zero_n_returns_nan(self):
        chi2, p = ab._chi_square(0, 0, 3, 5)
        assert math.isnan(chi2)
        assert math.isnan(p)

    def test_normal_case_returns_finite_values(self):
        chi2, p = ab._chi_square(8, 10, 2, 10)
        assert not math.isnan(chi2)
        assert 0 <= p <= 1


class TestBinomRate:
    def test_zero_n_returns_nan(self):
        rate, p = ab._binom_rate(0, 0)
        assert math.isnan(rate)
        assert math.isnan(p)

    def test_typical_case(self):
        rate, p = ab._binom_rate(7, 10)
        assert rate == pytest.approx(0.7)
        assert 0 <= p <= 1
