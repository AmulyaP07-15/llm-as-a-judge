from types import SimpleNamespace
from unittest.mock import patch

import pytest

import groq_utils


def _fake_response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class TestIsRateLimitError:
    def test_status_code_429(self):
        e = SimpleNamespace(status_code=429)
        assert groq_utils._is_rate_limit_error(e) is True

    def test_429_in_message(self):
        assert groq_utils._is_rate_limit_error(Exception("HTTP 429 error")) is True

    def test_rate_limit_text_in_message(self):
        assert groq_utils._is_rate_limit_error(Exception("Rate_Limit exceeded")) is True

    def test_unrelated_error_is_false(self):
        assert groq_utils._is_rate_limit_error(ValueError("bad request")) is False


class TestCallGroq:
    def test_returns_stripped_text_and_elapsed(self):
        with patch.object(groq_utils, "time") as mock_time, \
             patch.object(groq_utils.client.chat.completions, "create") as mock_create:
            mock_time.sleep.return_value = None
            mock_time.time.side_effect = [100.0, 100.2]
            mock_create.return_value = _fake_response("  hello there  ")

            text, elapsed = groq_utils.call_groq("llama", [{"role": "user", "content": "hi"}])

            assert text == "hello there"
            assert elapsed == pytest.approx(0.2)

    def test_passes_model_id_and_max_tokens(self):
        with patch.object(groq_utils, "time"), \
             patch.object(groq_utils.client.chat.completions, "create") as mock_create:
            mock_create.return_value = _fake_response("ok")

            groq_utils.call_groq("llama", [{"role": "user", "content": "hi"}], max_tokens=42)

            kwargs = mock_create.call_args.kwargs
            assert kwargs["model"] == groq_utils.MODEL_IDS["llama"]
            assert kwargs["max_tokens"] == 42
            assert "reasoning_effort" not in kwargs

    def test_qwen_gets_reasoning_effort_none(self):
        with patch.object(groq_utils, "time"), \
             patch.object(groq_utils.client.chat.completions, "create") as mock_create:
            mock_create.return_value = _fake_response("ok")

            groq_utils.call_groq("qwen", [{"role": "user", "content": "hi"}])

            assert mock_create.call_args.kwargs["reasoning_effort"] == "none"

    def test_retries_on_rate_limit_then_succeeds(self):
        with patch.object(groq_utils, "time"), \
             patch.object(groq_utils.client.chat.completions, "create") as mock_create:
            mock_create.side_effect = [Exception("429 too many requests"), _fake_response("ok")]

            text, _ = groq_utils.call_groq("llama", [{"role": "user", "content": "hi"}])

            assert text == "ok"
            assert mock_create.call_count == 2

    def test_raises_immediately_on_non_rate_limit_error(self):
        with patch.object(groq_utils, "time"), \
             patch.object(groq_utils.client.chat.completions, "create") as mock_create:
            mock_create.side_effect = ValueError("malformed request")

            with pytest.raises(ValueError):
                groq_utils.call_groq("llama", [{"role": "user", "content": "hi"}])

            assert mock_create.call_count == 1

    def test_raises_after_exhausting_retries(self):
        with patch.object(groq_utils, "time"), \
             patch.object(groq_utils.client.chat.completions, "create") as mock_create:
            mock_create.side_effect = Exception("429 rate_limit")

            with pytest.raises(Exception):
                groq_utils.call_groq("llama", [{"role": "user", "content": "hi"}])

            assert mock_create.call_count == groq_utils.MAX_RETRIES
