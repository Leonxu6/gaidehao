"""
Unit tests for src/pipeline/rewriter.py.

Uses the real DeepSeek API — no HTTP mocking.
The API key is injected via the autouse fixture.

Test text: same Chinese climate text used in M1/M2 tests.
"""

from __future__ import annotations

import os
import re

import pytest

# ---------------------------------------------------------------------------
# API key injection (must happen before any config/client imports)
# ---------------------------------------------------------------------------

DEEPSEEK_API_KEY = "sk-010552964cca453a92cb82756ca9baa0"


@pytest.fixture(autouse=True)
def set_api_key():
    """Inject the test API key into the environment for every test."""
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    yield


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.config import LLMConfig
from src.models.llm_client import LLMClient
from src.pipeline.proposition_extractor import Proposition
from src.pipeline.rewriter import ProbabilisticRewriter, RewriteConfig, RewriteResult

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

SAMPLE_PROPOSITIONS = [
    Proposition(text="气候变化是21世纪最严峻的全球性挑战之一", prop_type="claim", importance="high"),
    Proposition(text="过去100年全球平均气温上升了约1.1摄氏度", prop_type="fact", importance="high"),
    Proposition(text="中国已承诺在2060年前实现碳中和目标", prop_type="fact", importance="high"),
    Proposition(text="碳排放是气候变化的主要驱动力", prop_type="claim", importance="medium"),
    Proposition(text="IPCC报告", prop_type="entity", importance="medium"),
]

# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped to minimise API calls)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rewriter():
    """Create one ProbabilisticRewriter backed by the real DeepSeek API."""
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
    cfg = LLMConfig(api_key=DEEPSEEK_API_KEY)
    client = LLMClient(config=cfg)
    return ProbabilisticRewriter(llm_client=client)


@pytest.fixture(scope="module")
def chinese_result(rewriter):
    """Pre-computed rewrite of CHINESE_TEXT (no propositions) — reused by multiple tests."""
    return rewriter.rewrite(CHINESE_TEXT)


@pytest.fixture(scope="module")
def chinese_result_with_props(rewriter):
    """Pre-computed rewrite of CHINESE_TEXT with propositions."""
    return rewriter.rewrite(CHINESE_TEXT, propositions=SAMPLE_PROPOSITIONS)


@pytest.fixture(scope="module")
def english_result(rewriter):
    """Pre-computed rewrite of ENGLISH_TEXT."""
    return rewriter.rewrite(ENGLISH_TEXT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rewrite_returns_rewrite_result(chinese_result):
    """rewrite() must return a RewriteResult instance."""
    assert isinstance(chinese_result, RewriteResult), (
        f"Expected RewriteResult, got {type(chinese_result)}"
    )


def test_rewrite_text_not_empty(chinese_result):
    """rewritten_text must be a non-empty string."""
    assert isinstance(chinese_result.rewritten_text, str)
    assert chinese_result.rewritten_text.strip() != "", (
        "rewritten_text must not be empty"
    )


def test_rewrite_text_different_from_original(chinese_result):
    """The rewritten text must differ from the original."""
    assert chinese_result.rewritten_text != chinese_result.original_text, (
        "Rewritten text must differ from the original"
    )


def test_rewrite_with_propositions(chinese_result_with_props):
    """Passing propositions must not cause an error; result must be valid."""
    result = chinese_result_with_props
    assert isinstance(result, RewriteResult)
    assert result.rewritten_text.strip() != ""
    # propositions_included should contain the high-importance proposition texts
    high_texts = {p.text for p in SAMPLE_PROPOSITIONS if p.importance == "high"}
    assert set(result.propositions_included) == high_texts, (
        "propositions_included should contain only high-importance proposition texts"
    )


def test_rewrite_preserves_key_entities(chinese_result_with_props):
    """
    Key numerical entities from the source text must still be present in
    the rewrite (e.g. '1.1' for the temperature rise, '2060' for the target year).
    """
    rewritten = chinese_result_with_props.rewritten_text
    # '1.1' or '1.1摄氏度' — the specific degree figure
    assert re.search(r"1\.1", rewritten), (
        "Expected '1.1' (temperature rise) to appear in the rewritten text"
    )
    # '2060' — the carbon-neutrality commitment year
    assert "2060" in rewritten, (
        "Expected '2060' (carbon-neutrality year) to appear in the rewritten text"
    )


def test_rewrite_paragraph(rewriter):
    """rewrite_paragraph() must return a non-empty string."""
    para = "碳排放是气候变化的主要驱动力，工业化国家对此负有主要责任。"
    result = rewriter.rewrite_paragraph(para)
    assert isinstance(result, str), "rewrite_paragraph must return a str"
    assert result.strip() != "", "rewrite_paragraph must return a non-empty string"


def test_rewrite_attempt_2(rewriter):
    """attempt=2 (detection-evasion retry) must not raise and must return valid result."""
    result = rewriter.rewrite(CHINESE_TEXT, propositions=SAMPLE_PROPOSITIONS, attempt=2)
    assert isinstance(result, RewriteResult)
    assert result.rewritten_text.strip() != ""
    assert result.attempt_number == 2


def test_rewrite_english_text(english_result):
    """English input must produce a valid, non-empty RewriteResult."""
    assert isinstance(english_result, RewriteResult)
    assert english_result.rewritten_text.strip() != "", (
        "rewritten_text must not be empty for English input"
    )
    assert english_result.rewritten_text != english_result.original_text, (
        "Rewritten English text must differ from the original"
    )
