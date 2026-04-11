"""
Unit tests for src/models/llm_client.py.

These tests use the real DeepSeek API - no mocking of the HTTP layer.
They DO mock at the openai.OpenAI level for the error-handling test to
avoid actually sending a bad request.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

# API key - loaded from .env by tests/conftest.py
from tests.conftest import get_api_key

DEEPSEEK_API_KEY = get_api_key()


# Import after the key is injectable (config reads env at instantiation time)
from src.config import LLMConfig
from src.models.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client() -> LLMClient:
    cfg = LLMConfig(api_key=DEEPSEEK_API_KEY)
    return LLMClient(config=cfg)


# ---------------------------------------------------------------------------
# Test: generate() returns a non-empty string
# ---------------------------------------------------------------------------


def test_generate_returns_string():
    """generate() should return a non-empty string from the real API."""
    client = make_client()
    result = client.generate(
        prompt="Say exactly: hello world",
        system_prompt="You are a test assistant. Follow instructions precisely.",
        temperature=0.0,
        max_tokens=20,
    )
    assert isinstance(result, str), "generate() must return a str"
    assert len(result.strip()) > 0, "generate() must return non-empty text"


# ---------------------------------------------------------------------------
# Test: generate_json() returns a valid dict
# ---------------------------------------------------------------------------


def test_generate_json_returns_dict():
    """generate_json() should return a Python dict parsed from valid JSON."""
    client = make_client()
    result = client.generate_json(
        prompt=(
            "Return a JSON object with exactly two keys: "
            '"status" (string "ok") and "count" (integer 1).'
        ),
        system_prompt="You are a test assistant. Respond with valid JSON only.",
    )
    assert isinstance(result, dict), "generate_json() must return a dict"
    assert len(result) > 0, "Returned dict must not be empty"


# ---------------------------------------------------------------------------
# Test: generate_json() result has expected structure
# ---------------------------------------------------------------------------


def test_generate_json_structure():
    """generate_json() result should contain the keys we asked for."""
    client = make_client()
    result = client.generate_json(
        prompt=('Return JSON with keys "name" (string "ScholAI") and "version" (integer 1).'),
        system_prompt="You are a test assistant. Respond with valid JSON only.",
    )
    assert "name" in result or "version" in result, (
        "Expected at least one of 'name' or 'version' in the response"
    )


# ---------------------------------------------------------------------------
# Test: error handling - network failure triggers retries then raises
# ---------------------------------------------------------------------------


def test_generate_retries_on_connection_error():
    """
    When the API raises APIConnectionError, generate() should retry up to
    max_retries times and then re-raise the exception.
    """
    cfg = LLMConfig(
        api_key=DEEPSEEK_API_KEY, max_retries=3, retry_min_wait=0.01, retry_max_wait=0.05
    )
    client = LLMClient(config=cfg)

    call_count = 0

    def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        raise openai.APIConnectionError(request=MagicMock())

    with patch.object(client._client.chat.completions, "create", side_effect=fake_create):
        with pytest.raises(openai.APIConnectionError):
            client.generate(prompt="test", temperature=0.0, max_tokens=5)

    assert call_count == cfg.max_retries, f"Expected {cfg.max_retries} attempts, got {call_count}"


# ---------------------------------------------------------------------------
# Test: generate() with custom temperature override
# ---------------------------------------------------------------------------


def test_generate_with_temperature_override():
    """generate() should accept and honour a temperature override."""
    client = make_client()
    result = client.generate(
        prompt="What is 2 + 2? Answer with just the number.",
        temperature=0.0,
        max_tokens=10,
    )
    assert isinstance(result, str)
    assert "4" in result


# ---------------------------------------------------------------------------
# Test: streaming returns non-empty chunks
# ---------------------------------------------------------------------------


def test_generate_stream_yields_chunks():
    """generate_stream() should yield at least one non-empty string chunk."""
    client = make_client()
    chunks = list(
        client.generate_stream(
            prompt="Count from 1 to 3, separated by spaces.",
            temperature=0.0,
            max_tokens=20,
        )
    )
    assert len(chunks) > 0, "Stream should yield at least one chunk"
    full_text = "".join(chunks)
    assert len(full_text.strip()) > 0, "Streamed text must be non-empty"


# ---------------------------------------------------------------------------
# Test: _robust_json_parse edge cases
# ---------------------------------------------------------------------------


def test_robust_json_parse_strips_markdown_fence():
    """Strategy 2: markdown fences should be stripped before parsing."""
    raw = '```json\n{"status": "ok", "count": 1}\n```'
    result = LLMClient._robust_json_parse(raw)
    assert result == {"status": "ok", "count": 1}


def test_robust_json_parse_strips_markdown_fence_no_lang_tag():
    """Strategy 2: fences without 'json' language tag should also be stripped."""
    raw = '```\n{"key": "value"}\n```'
    result = LLMClient._robust_json_parse(raw)
    assert result == {"key": "value"}


def test_robust_json_parse_fullwidth_quotes():
    """Strategy 4: curly/fallback quotes normalised to ASCII."""
    # Note: fullwidth colon (\uff1a) followed by space produces {"key： "val"}
    # which is not valid JSON even after normalisation.  Strategy 4 normalises
    # fullwidth punctuation that appears INSIDE valid JSON structure.
    raw = '{"status": \u201cok\u201d, \u201cname\u201d: \u201ctest\u201d}'
    result = LLMClient._robust_json_parse(raw)
    assert result == {"status": "ok", "name": "test"}


def test_robust_json_parse_trailing_comma():
    """Strategy 4: trailing commas before ] or } should be removed."""
    raw = '{"items": [1, 2, 3,], "count": 3}'
    result = LLMClient._robust_json_parse(raw)
    assert result == {"items": [1, 2, 3], "count": 3}


def test_robust_json_parse_extracts_first_dict_block():
    """Strategy 3: first {...} block should be extracted when direct parse fails."""
    raw = 'Here is the result: {\"result\": \"success\"}. Thank you!'
    result = LLMClient._robust_json_parse(raw)
    assert result == {"result": "success"}


def test_robust_json_parse_item_level_recovery():
    """Strategy 5: item-level extraction should recover propositions from broken JSON."""
    raw = '{"propositions": [{"text": "Item one", "type": "fact"}, {"text": "Item two", "type": "claim"}]}'
    result = LLMClient._robust_json_parse(raw)
    assert "propositions" in result
    assert len(result["propositions"]) == 2


def test_robust_json_parse_valid_input_passthrough():
    """Well-formed JSON should parse on the first attempt (strategy 1)."""
    raw = '{"status": "ok", "count": 42}'
    result = LLMClient._robust_json_parse(raw)
    assert result == {"status": "ok", "count": 42}


def test_robust_json_parse_truly_invalid_raises():
    """Completely unparseable input should raise JSONDecodeError."""
    raw = "this is not json at all {{}}"
    import json
    with pytest.raises(json.JSONDecodeError):
        LLMClient._robust_json_parse(raw)
