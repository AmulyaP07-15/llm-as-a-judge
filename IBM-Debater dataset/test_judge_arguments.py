import csv
import random
from unittest.mock import patch

import pytest


@pytest.fixture
def jm(load_module):
    return load_module("IBM-Debater_judge_arguments.py")


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


class TestLoadArguments:
    def test_groups_by_topic_then_model(self, jm, tmp_path):
        path = tmp_path / "argument_outputs.csv"
        _write_csv(path, ["model", "topic", "argument"], [
            ["llama", "t1", "arg one"],
            ["qwen", "t1", "arg two"],
            ["llama", "t2", "arg three"],
        ])

        result = jm.load_arguments(str(path))

        assert dict(result) == {
            "t1": {"llama": "arg one", "qwen": "arg two"},
            "t2": {"llama": "arg three"},
        }

    def test_skips_error_rows(self, jm, tmp_path):
        path = tmp_path / "argument_outputs.csv"
        _write_csv(path, ["model", "topic", "argument"], [
            ["llama", "t1", "arg one"],
            ["qwen", "t1", "ERROR: rate limited"],
        ])

        result = jm.load_arguments(str(path))

        assert result["t1"] == {"llama": "arg one"}


class TestLoadCompletedKeys:
    def test_missing_file_returns_empty_set(self, jm, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        assert jm.load_completed_keys(str(missing)) == set()

    def test_empty_file_returns_empty_set(self, jm, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        assert jm.load_completed_keys(str(path)) == set()

    def test_reads_existing_keys(self, jm, tmp_path):
        path = tmp_path / "judge_verdicts.csv"
        _write_csv(
            path,
            ["judge_model", "topic", "model_a", "model_b", "swapped", "first_model",
             "second_model", "first_arg_length_words", "second_arg_length_words",
             "response_time_sec", "confidence", "raw_response", "winner_model"],
            [["llama", "t1", "allam", "qwen", "False", "allam", "qwen", "5", "6",
              "0.2", "4", "Winner: Argument 1", "allam"]],
        )

        keys = jm.load_completed_keys(str(path))

        assert keys == {("t1", "allam", "qwen", "llama")}


class TestBuildJudgePrompt:
    def test_contains_topic_and_both_arguments(self, jm):
        prompt = jm.build_judge_prompt("cats are great", "argument A", "argument B")

        assert "cats are great" in prompt
        assert "Argument 1: argument A" in prompt
        assert "Argument 2: argument B" in prompt
        assert "Winner:" in prompt
        assert "Confidence:" in prompt


class TestIsEchoedPrompt:
    @pytest.mark.parametrize("text", [
        "Winner: <Argument 1, Argument 2, or Neither>",
        "please answer with argument 1, argument 2, or neither",
    ])
    def test_detects_echoed_placeholder(self, jm, text):
        assert jm.is_echoed_prompt(text) is True

    def test_normal_verdict_is_not_echoed(self, jm):
        assert jm.is_echoed_prompt("Winner: Argument 1\nConfidence: 4") is False


class TestParseVerdict:
    def test_argument_1_wins(self, jm):
        choice, confidence = jm.parse_verdict("Winner: Argument 1\nConfidence: 4")
        assert choice == 1
        assert confidence == 4.0

    def test_argument_2_wins(self, jm):
        choice, confidence = jm.parse_verdict("Winner: Argument 2\nConfidence: 5")
        assert choice == 2
        assert confidence == 5.0

    def test_neither(self, jm):
        choice, confidence = jm.parse_verdict("Winner: Neither\nConfidence: 2")
        assert choice == "NEITHER"
        assert confidence == 2.0

    def test_decimal_confidence(self, jm):
        _, confidence = jm.parse_verdict("Winner: Argument 1\nConfidence: 4.5")
        assert confidence == 4.5

    def test_confidence_outside_scale_is_dropped(self, jm):
        _, confidence = jm.parse_verdict("Winner: Argument 1\nConfidence: 9")
        assert confidence is None

    def test_missing_winner_line_returns_none_choice(self, jm):
        choice, confidence = jm.parse_verdict("I think the first one is better.\nConfidence: 3")
        assert choice is None
        assert confidence == 3.0

    def test_missing_confidence_is_none(self, jm):
        choice, confidence = jm.parse_verdict("Winner: Argument 2")
        assert choice == 2
        assert confidence is None

    def test_unparseable_winner_text_returns_none(self, jm):
        choice, _ = jm.parse_verdict("Winner: I'm not sure\nConfidence: 3")
        assert choice is None

    def test_argument_1_substring_inside_longer_winner_text_wins_the_match(self, jm):
        choice, _ = jm.parse_verdict("Winner: Argument 2 is stronger than Argument 1")
        assert choice == 1


class TestGetVerdict:
    def test_calls_call_groq_with_built_prompt(self, jm):
        with patch.object(jm, "call_groq", return_value=("Winner: Argument 1", 0.3)) as mock_call:
            result = jm.get_verdict("llama", "topic", "arg a", "arg b")

        assert result == ("Winner: Argument 1", 0.3)
        model_name, messages = mock_call.call_args.args
        assert model_name == "llama"
        assert messages == [{"role": "user", "content": jm.build_judge_prompt("topic", "arg a", "arg b")}]
        assert mock_call.call_args.kwargs == {"max_tokens": 50}


class TestBuildJobs:
    def test_skips_topics_with_fewer_than_two_arguments(self, jm):
        by_topic = {"t1": {"llama": "a"}, "t2": {"llama": "a", "qwen": "b"}}

        random.seed(0)
        jobs = jm.build_jobs(by_topic, ["t1", "t2"])

        topics_present = {job[0] for job in jobs}
        assert topics_present == {"t2"}

    def test_every_pair_judged_by_every_judge_model(self, jm):
        by_topic = {"t1": {"llama": "a", "qwen": "b", "allam": "c"}}

        random.seed(0)
        jobs = jm.build_jobs(by_topic, ["t1"])

        # 3 models -> 3 pairwise combinations, each judged by all 3 judges
        assert len(jobs) == 3 * len(jm.JUDGE_MODELS)
        judges_seen = {job[3] for job in jobs}
        assert judges_seen == set(jm.JUDGE_MODELS)

    def test_all_judges_see_the_same_swap_for_a_given_pair(self, jm):
        by_topic = {"t1": {"llama": "a", "qwen": "b", "allam": "c"}}

        random.seed(0)
        jobs = jm.build_jobs(by_topic, ["t1"])

        swap_by_pair = {}
        for topic, model_a, model_b, judge, swapped in jobs:
            key = (topic, model_a, model_b)
            swap_by_pair.setdefault(key, swapped)
            assert swap_by_pair[key] == swapped

    def test_swap_flags_are_an_exact_half_split(self, jm):
        # 6 topics with 3 models each -> 6 pairs total, split evenly.
        by_topic = {
            f"t{i}": {"llama": "a", "qwen": "b", "allam": "c"} for i in range(6)
        }

        random.seed(0)
        jobs = jm.build_jobs(by_topic, list(by_topic.keys()))

        unique_pairs = {(topic, a, b): swapped for topic, a, b, _judge, swapped in jobs}
        swapped_count = sum(1 for v in unique_pairs.values() if v)
        assert swapped_count == len(unique_pairs) // 2


class TestMain:
    def _setup(self, jm, tmp_path, arguments_rows):
        args_file = tmp_path / "argument_outputs.csv"
        out_file = tmp_path / "judge_verdicts.csv"
        _write_csv(args_file, ["model", "topic", "argument"], arguments_rows)
        return args_file, out_file

    def test_writes_a_row_per_job_with_resolved_winner(self, jm, tmp_path):
        args_file, out_file = self._setup(jm, tmp_path, [
            ["llama", "t1", "arg llama"],
            ["qwen", "t1", "arg qwen"],
        ])

        with patch.object(jm, "ARGUMENTS_FILE", str(args_file)), \
             patch.object(jm, "OUTPUT_FILE", str(out_file)), \
             patch.object(jm, "JUDGE_MODELS", ["llama"]), \
             patch.object(jm, "get_verdict", return_value=("Winner: Argument 1\nConfidence: 4", 0.1)):
            jm.main()

        with open(out_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0][0] == "judge_model"
        assert len(rows) == 2
        data_row = dict(zip(rows[0], rows[1]))
        assert data_row["judge_model"] == "llama"
        assert data_row["winner_model"] in {"llama", "qwen"}

    def test_resumes_and_skips_already_completed_jobs(self, jm, tmp_path):
        args_file, out_file = self._setup(jm, tmp_path, [
            ["llama", "t1", "arg llama"],
            ["qwen", "t1", "arg qwen"],
        ])

        header = ["judge_model", "topic", "model_a", "model_b", "swapped", "first_model",
                  "second_model", "first_arg_length_words", "second_arg_length_words",
                  "response_time_sec", "confidence", "raw_response", "winner_model"]
        _write_csv(out_file, header, [
            ["llama", "t1", "llama", "qwen", "False", "llama", "qwen", "2", "2",
             "0.1", "4", "Winner: Argument 1\nConfidence: 4", "llama"],
        ])

        with patch.object(jm, "ARGUMENTS_FILE", str(args_file)), \
             patch.object(jm, "OUTPUT_FILE", str(out_file)), \
             patch.object(jm, "JUDGE_MODELS", ["llama"]), \
             patch.object(jm, "get_verdict") as mock_get_verdict:
            jm.main()

        # the only (topic, model_a, model_b, judge) job was already logged,
        # so no new judge calls should have been made
        mock_get_verdict.assert_not_called()

    def test_api_error_is_recorded_without_crashing(self, jm, tmp_path):
        args_file, out_file = self._setup(jm, tmp_path, [
            ["llama", "t1", "arg llama"],
            ["qwen", "t1", "arg qwen"],
        ])

        with patch.object(jm, "ARGUMENTS_FILE", str(args_file)), \
             patch.object(jm, "OUTPUT_FILE", str(out_file)), \
             patch.object(jm, "JUDGE_MODELS", ["llama"]), \
             patch.object(jm, "get_verdict", side_effect=RuntimeError("boom")):
            jm.main()

        with open(out_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        data_row = dict(zip(rows[0], rows[1]))
        assert data_row["winner_model"] == "API_ERROR"
        assert data_row["raw_response"].startswith("ERROR:")
