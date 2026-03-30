"""
Unit tests for src/pipeline/proposition_extractor.py.

Tests use the real DeepSeek API — no HTTP mocking.
The API key is injected via the autouse fixture below.
"""

from __future__ import annotations

import os
import pytest

# ---------------------------------------------------------------------------
# Inject the API key before any config/client imports
# ---------------------------------------------------------------------------

DEEPSEEK_API_KEY = "sk-010552964cca453a92cb82756ca9baa0"


@pytest.fixture(autouse=True)
def set_api_key():
    """Inject the test API key into the environment for every test."""
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    yield
    # No teardown needed — tests that create LLMConfig pass the key explicitly


# ---------------------------------------------------------------------------
# Imports (after key is injectable)
# ---------------------------------------------------------------------------

from src.config import LLMConfig
from src.models.llm_client import LLMClient
from src.pipeline.proposition_extractor import (
    ExtractionResult,
    Proposition,
    PropositionExtractor,
    _is_chinese,
)

# ---------------------------------------------------------------------------
# Test texts
# ---------------------------------------------------------------------------

CHINESE_TEXT = (
    "气候变化是21世纪最严峻的全球性挑战之一。根据IPCC的报告，过去100年全球平均气温上升了约1.1摄氏度。"
    "碳排放是气候变化的主要驱动力，工业化国家对此负有主要责任。中国作为世界最大碳排放国，"
    "已承诺在2060年前实现碳中和目标。"
)

ENGLISH_TEXT = (
    "Artificial intelligence has transformed numerous industries over the past decade. "
    "According to McKinsey, AI could contribute up to $13 trillion to global GDP by 2030. "
    "However, concerns about job displacement remain significant, with estimates suggesting "
    "that up to 800 million jobs could be automated by 2030."
)

SHORT_TEXT = "今天天气好"

# ---------------------------------------------------------------------------
# Shared fixture: one extractor instance per test module run
# (session scope avoids creating multiple OpenAI clients)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def extractor():
    """Create a PropositionExtractor backed by the real DeepSeek API."""
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    cfg = LLMConfig(api_key=DEEPSEEK_API_KEY)
    client = LLMClient(config=cfg)
    return PropositionExtractor(llm_client=client, config=cfg)


# ---------------------------------------------------------------------------
# Pre-compute results once to avoid redundant API calls for the Chinese tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def chinese_result(extractor):
    return extractor.extract(CHINESE_TEXT)


@pytest.fixture(scope="module")
def english_result(extractor):
    return extractor.extract(ENGLISH_TEXT)


@pytest.fixture(scope="module")
def short_result(extractor):
    return extractor.extract(SHORT_TEXT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_returns_extraction_result(chinese_result):
    """extract() must return an ExtractionResult instance."""
    assert isinstance(chinese_result, ExtractionResult), (
        f"Expected ExtractionResult, got {type(chinese_result)}"
    )


def test_extract_has_propositions(chinese_result):
    """The proposition list must be non-empty for a substantive Chinese text."""
    assert len(chinese_result.propositions) > 0, (
        "Expected at least one proposition from the Chinese climate text"
    )


def test_extract_proposition_has_required_fields(chinese_result):
    """Every Proposition must expose text, prop_type, and importance."""
    for prop in chinese_result.propositions:
        assert isinstance(prop, Proposition)
        assert hasattr(prop, "text"),       "Proposition missing 'text'"
        assert hasattr(prop, "prop_type"),  "Proposition missing 'prop_type'"
        assert hasattr(prop, "importance"), "Proposition missing 'importance'"


def test_extract_proposition_types_valid(chinese_result):
    """prop_type must be one of 'claim', 'entity', 'fact'."""
    valid_types = {"claim", "entity", "fact"}
    for prop in chinese_result.propositions:
        assert prop.prop_type in valid_types, (
            f"Invalid prop_type {prop.prop_type!r} — must be one of {valid_types}"
        )


def test_extract_importance_valid(chinese_result):
    """importance must be one of 'high', 'medium', 'low'."""
    valid_importances = {"high", "medium", "low"}
    for prop in chinese_result.propositions:
        assert prop.importance in valid_importances, (
            f"Invalid importance {prop.importance!r} — must be one of {valid_importances}"
        )


def test_extract_english_text(english_result):
    """English input must also produce a valid, non-empty ExtractionResult."""
    assert isinstance(english_result, ExtractionResult)
    assert len(english_result.propositions) > 0, (
        "Expected at least one proposition from the English AI text"
    )
    valid_types = {"claim", "entity", "fact"}
    valid_importances = {"high", "medium", "low"}
    for prop in english_result.propositions:
        assert prop.prop_type in valid_types
        assert prop.importance in valid_importances


def test_extract_no_empty_propositions(chinese_result, english_result):
    """No proposition should have an empty or whitespace-only text field."""
    for result in (chinese_result, english_result):
        for prop in result.propositions:
            assert prop.text.strip() != "", (
                "Found proposition with empty text field"
            )


def test_extract_short_text(short_result):
    """
    A very short text must not raise an exception.
    The result is a valid ExtractionResult; propositions may be empty or non-empty.
    """
    assert isinstance(short_result, ExtractionResult), (
        "extract() on short text must still return ExtractionResult"
    )
    # If propositions were returned, they must still be valid
    for prop in short_result.propositions:
        assert prop.text.strip() != ""
        assert prop.prop_type in {"claim", "entity", "fact"}
        assert prop.importance in {"high", "medium", "low"}
