"""
Integration tests for the full ScholAI pipeline (M7).

All six tests share a single module-scoped pipeline fixture to avoid
repeating the expensive model-loading and API calls for each test.

Tests
-----
1. test_pipeline_from_config         — from_config() returns a ScholAIPipeline
2. test_pipeline_run_returns_result  — run() returns a PipelineResult
3. test_pipeline_final_text_not_empty — final_text is a non-empty string
4. test_pipeline_final_text_differs_from_input — output != input
5. test_pipeline_result_has_metadata  — propositions_count >= 0, total_attempts >= 1
6. test_pipeline_warning_is_string   — warning field is a str (may be empty)
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Inject API key before any config imports
# ---------------------------------------------------------------------------
# API key is loaded automatically by tests/conftest.py (runs before this module)
from tests.conftest import get_api_key

os.environ["DEEPSEEK_API_KEY"] = get_api_key()

from src.pipeline import PipelineConfig, PipelineResult, ScholAIPipeline

# ---------------------------------------------------------------------------
# Test text — deliberately short to keep wall-clock time reasonable
# ---------------------------------------------------------------------------

SHORT_AI_TEXT = (
    "人工智能是当今最重要的技术之一。"
    "人工智能改变了很多行业。"
    "人工智能有很多应用场景。"
    "人工智能将继续发展。"
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures — built and run once for the entire module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline() -> ScholAIPipeline:
    """Build a ScholAIPipeline once for all tests in this module."""
    config = PipelineConfig(
        deepseek_api_key=os.environ["DEEPSEEK_API_KEY"],
        max_retries=3,
    )
    return ScholAIPipeline.from_config(config)


@pytest.fixture(scope="module")
def pipeline_result(pipeline: ScholAIPipeline) -> PipelineResult:
    """Run the pipeline once and share the result across all tests."""
    return pipeline.run(SHORT_AI_TEXT)


# ---------------------------------------------------------------------------
# Test 1 — factory method returns correct type
# ---------------------------------------------------------------------------


def test_pipeline_from_config():
    """ScholAIPipeline.from_config() should return a ScholAIPipeline instance."""
    config = PipelineConfig(
        deepseek_api_key=os.environ["DEEPSEEK_API_KEY"],
        max_retries=3,
    )
    instance = ScholAIPipeline.from_config(config)
    assert isinstance(instance, ScholAIPipeline), f"Expected ScholAIPipeline, got {type(instance)}"


# ---------------------------------------------------------------------------
# Test 2 — run() returns PipelineResult
# ---------------------------------------------------------------------------


def test_pipeline_run_returns_result(pipeline_result: PipelineResult):
    """pipeline.run() should return a PipelineResult instance."""
    assert isinstance(pipeline_result, PipelineResult), (
        f"Expected PipelineResult, got {type(pipeline_result)}"
    )


# ---------------------------------------------------------------------------
# Test 3 — final_text is non-empty
# ---------------------------------------------------------------------------


def test_pipeline_final_text_not_empty(pipeline_result: PipelineResult):
    """The final_text field must be a non-empty string."""
    assert isinstance(pipeline_result.final_text, str), "final_text should be a str"
    assert len(pipeline_result.final_text.strip()) > 0, "final_text must not be empty"


# ---------------------------------------------------------------------------
# Test 4 — output differs from input
# ---------------------------------------------------------------------------


def test_pipeline_final_text_differs_from_input(pipeline_result: PipelineResult):
    """The humanised output should not be identical to the original AI text."""
    assert pipeline_result.final_text.strip() != SHORT_AI_TEXT.strip(), (
        "final_text is identical to the input — the pipeline produced no rewrite"
    )


# ---------------------------------------------------------------------------
# Test 5 — metadata fields are populated
# ---------------------------------------------------------------------------


def test_pipeline_result_has_metadata(pipeline_result: PipelineResult):
    """propositions_count >= 0 and total_attempts >= 1."""
    assert isinstance(pipeline_result.propositions_count, int), (
        "propositions_count should be an int"
    )
    assert pipeline_result.propositions_count >= 0, "propositions_count must be non-negative"
    assert isinstance(pipeline_result.total_attempts, int), "total_attempts should be an int"
    assert pipeline_result.total_attempts >= 1, (
        "total_attempts must be at least 1 (pipeline must have tried at least once)"
    )


# ---------------------------------------------------------------------------
# Test 6 — warning is a string
# ---------------------------------------------------------------------------


def test_pipeline_warning_is_string(pipeline_result: PipelineResult):
    """The warning field must always be a str (empty string when no warning)."""
    assert isinstance(pipeline_result.warning, str), (
        f"warning should be a str, got {type(pipeline_result.warning)}"
    )
