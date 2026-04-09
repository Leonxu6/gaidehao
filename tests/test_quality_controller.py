"""
Unit tests for src/pipeline/quality_controller.py — M5 dual-track quality control.

These tests use the real DeepSeek API for G-Eval scoring.
The API key is injected via the `set_api_key` fixture.
"""

from __future__ import annotations

import os
import pytest

# ---------------------------------------------------------------------------
# API key — loaded from .env by tests/conftest.py
from tests.conftest import get_api_key

DEEPSEEK_API_KEY = get_api_key()

# ---------------------------------------------------------------------------
# Imports (after the env var is injectable)
# ---------------------------------------------------------------------------

from src.config import LLMConfig
from src.models.llm_client import LLMClient
from src.pipeline.quality_controller import (
    QualityController,
    QualityReport,
    DetectionMetrics,
    QualityMetrics,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Simulates human-like writing: varied sentence lengths, rich vocabulary
HUMAN_LIKE_TEXT = """气候变化的威胁远比我们想象的更为紧迫。科学家们已经发出了明确警告——如果我们不采取行动，后果将是灾难性的。

过去一个世纪，全球平均气温上升了约1.1摄氏度。这个数字看似不大，却足以引发极端天气事件的急剧增加。干旱、洪水、飓风——这些灾害正在变得越来越频繁。

解决方案其实并不神秘。可再生能源已经足够便宜，太阳能和风能的成本在十年内下降了超过九成。问题不在于技术，而在于政治意愿。"""

# Simulates AI-generated writing: uniform sentence structure, repetitive vocabulary
AI_LIKE_TEXT = (
    "气候变化是一个重要的全球性问题。"
    "气候变化对环境有很大影响。"
    "气候变化需要全球合作来解决。"
    "气候变化的原因是碳排放增加。"
    "气候变化的后果是温度升高。"
)

# Text with a duplicate sentence for rule-check testing
REPETITIVE_TEXT = (
    "气候变化是一个重要的全球性问题。"
    "科学家们一直在研究这个问题。"
    "科学家们一直在研究这个问题。"  # deliberate duplicate
    "我们需要采取行动来应对气候变化。"
)


@pytest.fixture(scope="module")
def controller() -> QualityController:
    """Create a shared QualityController instance (one LLM client per module)."""
    cfg = LLMConfig(api_key=DEEPSEEK_API_KEY)
    client = LLMClient(config=cfg)
    return QualityController(llm_client=client)


@pytest.fixture(scope="module")
def human_report(controller: QualityController) -> QualityReport:
    """Evaluate HUMAN_LIKE_TEXT once and reuse across tests that need it."""
    return controller.evaluate(HUMAN_LIKE_TEXT)


# ---------------------------------------------------------------------------
# Test 1: evaluate() returns a QualityReport
# ---------------------------------------------------------------------------

def test_evaluate_returns_quality_report(human_report: QualityReport):
    """evaluate() must return a QualityReport instance."""
    assert isinstance(human_report, QualityReport)


# ---------------------------------------------------------------------------
# Test 2: DetectionMetrics structure
# ---------------------------------------------------------------------------

def test_detection_metrics_structure(human_report: QualityReport):
    """DetectionMetrics must expose ppl_estimate, burstiness, ttr, overall_passed."""
    d = human_report.detection
    assert isinstance(d, DetectionMetrics)
    assert hasattr(d, "ppl_estimate")
    assert hasattr(d, "burstiness")
    assert hasattr(d, "ttr")
    assert hasattr(d, "overall_passed")
    # Values must be non-negative floats
    assert d.ppl_estimate >= 0.0
    assert d.burstiness >= 0.0
    assert 0.0 <= d.ttr <= 1.0


# ---------------------------------------------------------------------------
# Test 3: QualityMetrics structure
# ---------------------------------------------------------------------------

def test_quality_metrics_structure(human_report: QualityReport):
    """QualityMetrics must expose rule_issues, faithfulness, coherence, fluency,
    readability, and overall_score."""
    q = human_report.quality
    assert isinstance(q, QualityMetrics)
    assert hasattr(q, "rule_issues")
    assert isinstance(q.rule_issues, list)
    assert hasattr(q, "faithfulness")
    assert hasattr(q, "coherence")
    assert hasattr(q, "fluency")
    assert hasattr(q, "readability")
    assert hasattr(q, "overall_score")


# ---------------------------------------------------------------------------
# Test 4: human-like text should have meaningful burstiness
# ---------------------------------------------------------------------------

def test_human_like_text_burstiness(controller: QualityController):
    """HUMAN_LIKE_TEXT has varying sentence lengths — burstiness should be >= 5."""
    metrics = controller._compute_detection_metrics(HUMAN_LIKE_TEXT)
    assert metrics.burstiness >= 5, (
        f"Expected burstiness >= 5 for human-like text, got {metrics.burstiness:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 5: AI-like text has low TTR (high lexical repetition)
# ---------------------------------------------------------------------------

def test_ai_like_text_ttr_low(controller: QualityController):
    """AI_LIKE_TEXT repeats '气候变化' heavily — its TTR should be lower than
    HUMAN_LIKE_TEXT, and concretely below 0.70 with character-level CJK tokenisation."""
    ai_metrics = controller._compute_detection_metrics(AI_LIKE_TEXT)
    human_metrics = controller._compute_detection_metrics(HUMAN_LIKE_TEXT)
    # Verify the relative ordering: AI text has lower vocabulary diversity
    assert ai_metrics.ttr < human_metrics.ttr, (
        f"Expected AI TTR ({ai_metrics.ttr:.4f}) < human TTR ({human_metrics.ttr:.4f})"
    )
    # Verify a concrete upper bound appropriate for character-level CJK tokenisation
    assert ai_metrics.ttr < 0.70, (
        f"Expected AI-like text TTR < 0.70, got {ai_metrics.ttr:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: G-Eval scores must be in [1, 5]
# ---------------------------------------------------------------------------

def test_geval_scores_in_range(human_report: QualityReport):
    """All four G-Eval dimensions must be in the valid range [1, 5]."""
    q = human_report.quality
    for name, value in [
        ("faithfulness", q.faithfulness),
        ("coherence", q.coherence),
        ("fluency", q.fluency),
        ("readability", q.readability),
    ]:
        assert 1.0 <= value <= 5.0, (
            f"G-Eval dimension '{name}' = {value} is outside [1, 5]"
        )


# ---------------------------------------------------------------------------
# Test 7: overall_score is correctly weighted
# ---------------------------------------------------------------------------

def test_overall_score_weighted(human_report: QualityReport):
    """overall_score must equal 0.35*f + 0.25*c + 0.25*fl + 0.15*r."""
    q = human_report.quality
    expected = (
        0.35 * q.faithfulness
        + 0.25 * q.coherence
        + 0.25 * q.fluency
        + 0.15 * q.readability
    )
    assert abs(q.overall_score - expected) < 1e-9, (
        f"overall_score {q.overall_score:.6f} != expected {expected:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8: rule check detects repetitive sentences
# ---------------------------------------------------------------------------

def test_rule_check_detects_repetition(controller: QualityController):
    """Text with duplicate consecutive sentences must produce rule_passed=False."""
    rule_passed, issues = controller._rule_check(REPETITIVE_TEXT)
    assert rule_passed is False, "Expected rule_passed=False for text with duplicate sentences"
    assert "重复句子" in issues, f"Expected '重复句子' in issues, got: {issues}"


# ---------------------------------------------------------------------------
# Test 9: final_action is one of the valid values
# ---------------------------------------------------------------------------

def test_final_action_valid(human_report: QualityReport):
    """final_action must be one of the three valid string values."""
    valid_actions = {"accept", "retry_detection", "retry_quality", "retry_semantic"}
    assert human_report.final_action in valid_actions, (
        f"final_action '{human_report.final_action}' is not a valid action"
    )
