"""
Unit tests for src/pipeline/semantic_anchor.py  (M2 — Semantic Anchor Embedding).

These tests use the real local BAAI/bge-m3 model (lazily loaded).
The model is ~570 MB and will be downloaded from HuggingFace Hub on first
run (cached in ~/.cache/huggingface/hub afterwards).

API key injection
-----------------
Even though this module does not call the DeepSeek LLM, importing src.config
reads DEEPSEEK_API_KEY at module level.  We inject a dummy value here so the
import succeeds in environments without a .env file.
"""

from __future__ import annotations

import os

# Must be set before any src.* import that transitively loads src.config
# API key is loaded automatically by tests/conftest.py (runs before this module)
from tests.conftest import get_api_key
os.environ["DEEPSEEK_API_KEY"] = get_api_key()

import numpy as np
import pytest

from src.models.embedding_client import EmbeddingClient
from src.pipeline.semantic_anchor import SemanticAnchor, SemanticAnchorBuilder

# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

MULTI_PARA_TEXT = """气候变化是21世纪最严峻的全球性挑战之一。根据IPCC的报告，过去100年全球平均气温上升了约1.1摄氏度。

碳排放是气候变化的主要驱动力，工业化国家对此负有主要责任。中国作为世界最大碳排放国，已承诺在2060年前实现碳中和目标。

可再生能源是解决气候变化的关键途径，太阳能和风能成本在过去十年降低了90%以上。"""

UNRELATED_TEXT = "今天股市大涨，科技股领涨，投资者情绪高涨，交易量创历史新高。"

SIMILAR_TEXT = """气候变化是当前全球面临的最严峻挑战。IPCC报告显示，过去一个世纪地球气温上升约1.1度。

工业化国家的碳排放是气候变化的核心驱动因素。中国承诺于2060年实现碳中和。"""


# ---------------------------------------------------------------------------
# Session-scoped fixtures (load the model only once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedding_client() -> EmbeddingClient:
    """One EmbeddingClient for the whole test session."""
    return EmbeddingClient()


@pytest.fixture(scope="session")
def builder(embedding_client: EmbeddingClient) -> SemanticAnchorBuilder:
    """One SemanticAnchorBuilder for the whole test session (default threshold 0.80)."""
    return SemanticAnchorBuilder(embedding_client)


@pytest.fixture(scope="session")
def multi_para_anchor(builder: SemanticAnchorBuilder) -> SemanticAnchor:
    """Anchor built from MULTI_PARA_TEXT, computed once per session."""
    return builder.build(MULTI_PARA_TEXT)


# ---------------------------------------------------------------------------
# Test 1: build() returns SemanticAnchor
# ---------------------------------------------------------------------------

def test_build_returns_semantic_anchor(multi_para_anchor: SemanticAnchor):
    """build() must return a SemanticAnchor instance."""
    assert isinstance(multi_para_anchor, SemanticAnchor), (
        f"Expected SemanticAnchor, got {type(multi_para_anchor)}"
    )


# ---------------------------------------------------------------------------
# Test 2: embedding is a 1-D ndarray with dim 1024
# ---------------------------------------------------------------------------

def test_anchor_embedding_shape(multi_para_anchor: SemanticAnchor):
    """anchor.embedding must be a 1-D ndarray with shape[0] == 1024."""
    emb = multi_para_anchor.embedding
    assert isinstance(emb, np.ndarray), "embedding must be np.ndarray"
    assert emb.ndim == 1, f"Expected 1-D embedding, got shape {emb.shape}"
    assert emb.shape[0] == 1024, (
        f"BGE-M3 output dim must be 1024, got {emb.shape[0]}"
    )


# ---------------------------------------------------------------------------
# Test 3: original_text is preserved verbatim
# ---------------------------------------------------------------------------

def test_anchor_original_text_preserved(builder: SemanticAnchorBuilder):
    """anchor.original_text must equal the text passed to build()."""
    text = "这是一个简单的测试文本。"
    anchor = builder.build(text)
    assert anchor.original_text == text, (
        "original_text must be stored verbatim without modification"
    )


# ---------------------------------------------------------------------------
# Test 4: multi-paragraph text produces >= 2 chunk embeddings
# ---------------------------------------------------------------------------

def test_anchor_has_chunks(multi_para_anchor: SemanticAnchor):
    """Multi-paragraph text (split on \\n\\n) must yield >= 2 chunk_embeddings."""
    assert len(multi_para_anchor.chunk_embeddings) >= 2, (
        f"Expected >= 2 chunk embeddings, got {len(multi_para_anchor.chunk_embeddings)}"
    )
    assert len(multi_para_anchor.chunks) == len(multi_para_anchor.chunk_embeddings), (
        "chunks and chunk_embeddings must have the same length"
    )


# ---------------------------------------------------------------------------
# Test 5: same text → similarity >= 0.99, passed=True
# ---------------------------------------------------------------------------

def test_check_similarity_same_text(builder: SemanticAnchorBuilder):
    """Comparing original text with itself should yield similarity >= 0.99."""
    anchor = builder.build(MULTI_PARA_TEXT)
    result = builder.check_similarity(anchor, MULTI_PARA_TEXT)
    assert result["similarity"] >= 0.99, (
        f"Self-similarity should be >= 0.99, got {result['similarity']:.4f}"
    )
    assert result["passed"] is True, "Self-comparison must pass the threshold"


# ---------------------------------------------------------------------------
# Test 6: completely unrelated text → passed=False (similarity < 0.80)
# ---------------------------------------------------------------------------

def test_check_similarity_different_text(builder: SemanticAnchorBuilder):
    """Comparing climate text with stock-market text should fail the 0.80 threshold."""
    anchor = builder.build(MULTI_PARA_TEXT)
    result = builder.check_similarity(anchor, UNRELATED_TEXT)
    assert result["similarity"] < 0.80, (
        f"Unrelated text similarity should be < 0.80, got {result['similarity']:.4f}"
    )
    assert result["passed"] is False, (
        "Unrelated text must not pass the coarse semantic threshold"
    )


# ---------------------------------------------------------------------------
# Test 7: check_similarity returns dict with required keys
# ---------------------------------------------------------------------------

def test_check_similarity_returns_dict_keys(builder: SemanticAnchorBuilder):
    """check_similarity() must return a dict with keys: similarity, passed, reason."""
    anchor = builder.build(MULTI_PARA_TEXT)
    result = builder.check_similarity(anchor, SIMILAR_TEXT)
    assert isinstance(result, dict), "Result must be a dict"
    for key in ("similarity", "passed", "reason"):
        assert key in result, f"Key '{key}' missing from result dict"
    assert isinstance(result["similarity"], float), "similarity must be float"
    assert isinstance(result["passed"], bool), "passed must be bool"
    assert isinstance(result["reason"], str), "reason must be str"


# ---------------------------------------------------------------------------
# Test 8: check_chunk_similarity returns correct structure
# ---------------------------------------------------------------------------

def test_check_chunk_similarity_structure(
    builder: SemanticAnchorBuilder,
    multi_para_anchor: SemanticAnchor,
):
    """check_chunk_similarity() must return a list of dicts with required keys."""
    results = builder.check_chunk_similarity(multi_para_anchor, SIMILAR_TEXT)
    assert isinstance(results, list), "Result must be a list"
    assert len(results) > 0, "Result list must not be empty"

    required_keys = {"chunk_idx", "original", "similarity", "passed"}
    for i, item in enumerate(results):
        assert isinstance(item, dict), f"Item {i} must be a dict"
        assert required_keys.issubset(item.keys()), (
            f"Item {i} missing keys: {required_keys - item.keys()}"
        )
        assert isinstance(item["chunk_idx"], int), f"chunk_idx must be int (item {i})"
        assert isinstance(item["original"], str), f"original must be str (item {i})"
        assert isinstance(item["similarity"], float), f"similarity must be float (item {i})"
        assert isinstance(item["passed"], bool), f"passed must be bool (item {i})"
        assert item["chunk_idx"] == i, (
            f"chunk_idx must equal list position: expected {i}, got {item['chunk_idx']}"
        )


# ---------------------------------------------------------------------------
# Test 9: custom threshold=0.95 — similar (but not identical) text must fail
# ---------------------------------------------------------------------------

def test_similarity_threshold_custom(embedding_client: EmbeddingClient):
    """
    With threshold=0.95, SIMILAR_TEXT (a paraphrase, not a copy) should not pass,
    while the original text compared with itself always passes at any threshold <= 1.0.
    This verifies that the custom threshold is actually applied.
    """
    strict_builder = SemanticAnchorBuilder(embedding_client, similarity_threshold=0.95)
    anchor = strict_builder.build(MULTI_PARA_TEXT)

    # Paraphrased (similar) text should fail at the strict 0.95 threshold
    result = strict_builder.check_similarity(anchor, SIMILAR_TEXT)
    assert result["passed"] is False, (
        f"With threshold=0.95, paraphrased text must fail. "
        f"Got similarity={result['similarity']:.4f}, passed={result['passed']}"
    )

    # Also verify the stored threshold on the anchor itself
    assert anchor.similarity_threshold == 0.95, (
        f"anchor.similarity_threshold should be 0.95, got {anchor.similarity_threshold}"
    )
