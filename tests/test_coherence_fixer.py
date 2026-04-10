"""
Unit tests for src/pipeline/coherence_fixer.py — M6 coherence fixer + retry orchestrator.

CoherenceFixer tests (no API required):
  1. test_fix_returns_string
  2. test_fix_single_paragraph_unchanged
  3. test_detect_incoherence_contrast_word
  4. test_detect_incoherence_coherent_paras

RetryOrchestrator tests (real API, short text):
  5. test_orchestrator_returns_result
  6. test_orchestrator_result_has_text
  7. test_orchestrator_attempts_recorded
  8. test_orchestrator_total_attempts_consistent
"""

from __future__ import annotations

import pytest

# API key — loaded from .env by tests/conftest.py
from tests.conftest import get_api_key

DEEPSEEK_API_KEY = get_api_key()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.config import EmbeddingConfig, LLMConfig
from src.models.embedding_client import EmbeddingClient
from src.models.llm_client import LLMClient
from src.pipeline.coherence_fixer import (
    AttemptRecord,
    CoherenceFixer,
    OrchestratorResult,
    RetryOrchestrator,
)
from src.pipeline.proposition_extractor import Proposition
from src.pipeline.quality_controller import QualityController
from src.pipeline.rewriter import ProbabilisticRewriter
from src.pipeline.semantic_anchor import SemanticAnchorBuilder
from src.pipeline.semantic_validator import SemanticValidator

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TEST_TEXT = (
    "气候变化是21世纪最严峻的全球性挑战。"
    "碳排放导致全球气温上升，极端天气增多。"
    "各国需要合作减少排放，推广可再生能源。"
)

TEST_PROPS = [Proposition("气候变化是21世纪最严峻的全球性挑战", "claim", "high")]

# ---------------------------------------------------------------------------
# Module-scoped shared components (built once, reused across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llm_client() -> LLMClient:
    cfg = LLMConfig(api_key=DEEPSEEK_API_KEY)
    return LLMClient(config=cfg)


@pytest.fixture(scope="module")
def rewriter(llm_client: LLMClient) -> ProbabilisticRewriter:
    return ProbabilisticRewriter(llm_client=llm_client)


@pytest.fixture(scope="module")
def coherence_fixer(rewriter: ProbabilisticRewriter) -> CoherenceFixer:
    return CoherenceFixer(rewriter=rewriter)


@pytest.fixture(scope="module")
def orchestrator(
    rewriter: ProbabilisticRewriter,
    llm_client: LLMClient,
    coherence_fixer: CoherenceFixer,
) -> RetryOrchestrator:
    embedding_cfg = EmbeddingConfig()
    embedding_client = EmbeddingClient(config=embedding_cfg)
    anchor_builder = SemanticAnchorBuilder(embedding_client=embedding_client)
    validator = SemanticValidator(anchor_builder=anchor_builder)
    quality_controller = QualityController(llm_client=llm_client)
    return RetryOrchestrator(
        rewriter=rewriter,
        validator=validator,
        quality_controller=quality_controller,
        coherence_fixer=coherence_fixer,
        max_retries=3,
        min_quality_score=3.0,
    )


@pytest.fixture(scope="module")
def anchor(orchestrator: RetryOrchestrator):
    """Build a SemanticAnchor from TEST_TEXT once per module."""
    return orchestrator._validator._anchor_builder.build(TEST_TEXT)


@pytest.fixture(scope="module")
def orchestrator_result(
    orchestrator: RetryOrchestrator,
    anchor,
) -> OrchestratorResult:
    """Run the orchestrator once and share the result across the four API tests."""
    return orchestrator.run(
        text=TEST_TEXT,
        anchor=anchor,
        propositions=TEST_PROPS,
    )


# ===========================================================================
# CoherenceFixer tests (no API calls)
# ===========================================================================


def test_fix_returns_string(coherence_fixer: CoherenceFixer):
    """fix() must return a non-empty string."""
    result = coherence_fixer.fix(TEST_TEXT)
    assert isinstance(result, str)
    assert len(result) > 0


def test_fix_single_paragraph_unchanged(coherence_fixer: CoherenceFixer):
    """A single-paragraph text must be returned as-is (no API call triggered)."""
    single_para = "气候变化是全球最重要的环境问题之一，各国都需要积极应对。"
    result = coherence_fixer.fix(single_para)
    assert result == single_para


def test_detect_incoherence_contrast_word(coherence_fixer: CoherenceFixer):
    """
    para2 starts with '但是' while para1 has no contrast-setup phrase.
    _detect_incoherence() must return True.
    """
    para1 = "可再生能源在过去十年间成本大幅下降，已经成为最具竞争力的电力来源。"
    para2 = "但是化石燃料仍然主导着全球能源供应，短期内很难被完全取代。"
    assert coherence_fixer._detect_incoherence(para1, para2) is True


def test_detect_incoherence_coherent_paras(coherence_fixer: CoherenceFixer):
    """
    Two clearly connected paragraphs must NOT be flagged as incoherent.
    para2 continues the same topic with similar sentence lengths and no
    unexplained contrast opener.
    """
    para1 = "气候变化正在对农业生产产生深远影响。干旱和洪涝的频率显著增加，威胁粮食安全。"
    para2 = "科学家们预测，到本世纪中叶，全球主要粮食产区的产量可能下降10%到25%。"
    assert coherence_fixer._detect_incoherence(para1, para2) is False


# ===========================================================================
# RetryOrchestrator tests (require API)
# ===========================================================================


def test_orchestrator_returns_result(orchestrator_result: OrchestratorResult):
    """run() must return an OrchestratorResult instance."""
    assert isinstance(orchestrator_result, OrchestratorResult)


def test_orchestrator_result_has_text(orchestrator_result: OrchestratorResult):
    """final_text must be a non-empty string."""
    assert isinstance(orchestrator_result.final_text, str)
    assert len(orchestrator_result.final_text) > 0


def test_orchestrator_attempts_recorded(orchestrator_result: OrchestratorResult):
    """attempts list must contain at least one AttemptRecord."""
    assert isinstance(orchestrator_result.attempts, list)
    assert len(orchestrator_result.attempts) >= 1
    for record in orchestrator_result.attempts:
        assert isinstance(record, AttemptRecord)


def test_orchestrator_total_attempts_consistent(orchestrator_result: OrchestratorResult):
    """total_attempts must equal the length of the attempts list."""
    assert orchestrator_result.total_attempts == len(orchestrator_result.attempts)
