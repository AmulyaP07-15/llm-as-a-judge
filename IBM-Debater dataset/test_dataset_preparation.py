import csv
from unittest.mock import patch

import pytest


@pytest.fixture
def prep(load_module):
    return load_module("IBM-Debater_dataset_preparation.py")


class TestLoadTopics:
    def test_dedupes_preserving_first_seen_order(self, prep):
        fake_dataset = {"topic": ["B", "A", "B", "C", "A"]}
        with patch.object(prep, "load_dataset", return_value=fake_dataset) as mock_load:
            topics = prep.load_topics("some/dataset")

        assert topics == ["B", "A", "C"]
        mock_load.assert_called_once_with("some/dataset", split="train")

    def test_uses_default_dataset_name(self, prep):
        with patch.object(prep, "load_dataset", return_value={"topic": []}) as mock_load:
            prep.load_topics()

        assert mock_load.call_args.args[0] == prep.DATASET_NAME


class TestSaveTopics:
    def test_writes_header_and_one_row_per_topic(self, prep, tmp_path):
        out_file = tmp_path / "topics.csv"

        prep.save_topics(["topic one", "topic two"], output_file=str(out_file))

        with open(out_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows == [["topic"], ["topic one"], ["topic two"]]

    def test_empty_list_still_writes_header(self, prep, tmp_path):
        out_file = tmp_path / "topics.csv"

        prep.save_topics([], output_file=str(out_file))

        with open(out_file, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows == [["topic"]]


class TestMain:
    def test_loads_then_saves_the_loaded_topics(self, prep):
        # save_topics' output_file default is bound at def-time to the
        # module's original TOPICS_OUTPUT_FILE, so patching that module
        # attribute wouldn't reach it -- assert the wiring instead of the
        # actual file write, which test_save_topics already covers.
        with patch.object(prep, "load_topics", return_value=["A", "B"]) as mock_load, \
             patch.object(prep, "save_topics") as mock_save:
            prep.main()

        mock_load.assert_called_once()
        mock_save.assert_called_once_with(["A", "B"])
