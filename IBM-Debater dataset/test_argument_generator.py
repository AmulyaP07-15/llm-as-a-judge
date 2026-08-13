import csv
from unittest.mock import patch

import pytest


@pytest.fixture
def gen(load_module):
    return load_module("IBM-Debater_dataset_argument_generator.py")


class TestLoadTopics:
    def _write_csv(self, path, topics):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["topic"])
            for t in topics:
                writer.writerow([t])

    def test_dedupes_preserving_first_seen_order(self, gen, tmp_path):
        path = tmp_path / "topics.csv"
        self._write_csv(path, ["B", "A", "B", "C"])

        assert gen.load_topics(str(path)) == ["B", "A", "C"]

    def test_limit_truncates_after_dedup(self, gen, tmp_path):
        path = tmp_path / "topics.csv"
        self._write_csv(path, ["A", "B", "C", "D"])

        assert gen.load_topics(str(path), limit=2) == ["A", "B"]

    def test_no_limit_returns_everything(self, gen, tmp_path):
        path = tmp_path / "topics.csv"
        self._write_csv(path, ["A", "B"])

        assert gen.load_topics(str(path), limit=None) == ["A", "B"]


class TestGetArgument:
    def test_builds_expected_prompt_and_returns_text_only(self, gen):
        with patch.object(gen, "call_groq", return_value=("the argument", 0.5)) as mock_call:
            result = gen.get_argument("llama", "cats are great")

        assert result == "the argument"
        model_name, messages = mock_call.call_args.args
        assert model_name == "llama"
        assert messages == [{
            "role": "user",
            "content": "Write a short 2-3 sentence argument supporting: cats are great",
        }]
        assert mock_call.call_args.kwargs == {"max_tokens": 150}


class TestMain:
    def test_writes_one_row_per_topic_model_pair(self, gen, tmp_path):
        out_file = tmp_path / "argument_outputs.csv"
        with patch.object(gen, "load_topics", return_value=["topic1", "topic2"]), \
             patch.object(gen, "MODELS", ["llama", "qwen"]), \
             patch.object(gen, "OUTPUT_FILE", str(out_file)), \
             patch.object(gen, "get_argument", side_effect=lambda model, topic: f"{model}-arg-for-{topic}"):
            gen.main()

        with open(out_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0] == ["model", "topic", "argument"]
        assert rows[1:] == [
            ["llama", "topic1", "llama-arg-for-topic1"],
            ["qwen", "topic1", "qwen-arg-for-topic1"],
            ["llama", "topic2", "llama-arg-for-topic2"],
            ["qwen", "topic2", "qwen-arg-for-topic2"],
        ]

    def test_records_error_string_instead_of_crashing(self, gen, tmp_path):
        out_file = tmp_path / "argument_outputs.csv"

        def flaky_get_argument(model, topic):
            if model == "qwen":
                raise RuntimeError("boom")
            return "ok"

        with patch.object(gen, "load_topics", return_value=["topic1"]), \
             patch.object(gen, "MODELS", ["llama", "qwen"]), \
             patch.object(gen, "OUTPUT_FILE", str(out_file)), \
             patch.object(gen, "get_argument", side_effect=flaky_get_argument):
            gen.main()

        with open(out_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[1] == ["llama", "topic1", "ok"]
        assert rows[2][:2] == ["qwen", "topic1"]
        assert rows[2][2].startswith("ERROR:")
