"""
Unit tests for src/models/embedding_client.py.

These tests use the real local BAAI/bge-m3 model.
The model is ~570 MB and will be downloaded from HuggingFace Hub on first run
(cached in ~/.cache/huggingface/hub afterwards).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import EmbeddingConfig
from src.models.embedding_client import EmbeddingClient

# ---------------------------------------------------------------------------
# Shared fixture: one client instance per test session (model is expensive to load)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def client() -> EmbeddingClient:
    """Return a single EmbeddingClient loaded once for the whole test session."""
    cfg = EmbeddingConfig(model_name="BAAI/bge-m3", device="cpu")
    return EmbeddingClient(config=cfg)


# ---------------------------------------------------------------------------
# Test: lazy loading — model should NOT be loaded before first encode() call
# ---------------------------------------------------------------------------


def test_lazy_loading_not_loaded_on_init():
    """Model should not be loaded at construction time."""
    cfg = EmbeddingConfig(model_name="BAAI/bge-m3", device="cpu")
    fresh_client = EmbeddingClient(config=cfg)
    assert not fresh_client.is_loaded, "Model should not be loaded at __init__ time (lazy loading)"


def test_lazy_loading_loaded_after_encode(client: EmbeddingClient):
    """Model should be loaded after the first encode() call."""
    _ = client.encode("trigger load")
    assert client.is_loaded, "Model should be loaded after encode() is called"


# ---------------------------------------------------------------------------
# Test: encode() returns correct shape and dtype
# ---------------------------------------------------------------------------


def test_encode_returns_ndarray(client: EmbeddingClient):
    """encode() should return a numpy ndarray."""
    emb = client.encode("Hello world")
    assert isinstance(emb, np.ndarray), "encode() must return np.ndarray"


def test_encode_returns_1d_vector(client: EmbeddingClient):
    """encode() should return a 1-D vector."""
    emb = client.encode("Hello world")
    assert emb.ndim == 1, f"Expected 1-D array, got shape {emb.shape}"


def test_encode_correct_dimension(client: EmbeddingClient):
    """BGE-M3 produces 1024-dimensional embeddings."""
    emb = client.encode("Hello world")
    assert emb.shape[0] == 1024, f"Expected embedding dim 1024, got {emb.shape[0]}"


def test_encode_dtype_float32(client: EmbeddingClient):
    """encode() should return float32 embeddings."""
    emb = client.encode("Hello world")
    assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"


# ---------------------------------------------------------------------------
# Test: encode_batch() returns correct shape
# ---------------------------------------------------------------------------


def test_encode_batch_shape(client: EmbeddingClient):
    """encode_batch() should return shape (N, embedding_dim)."""
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embs = client.encode_batch(texts)
    assert embs.shape == (3, 1024), f"Expected shape (3, 1024), got {embs.shape}"


def test_encode_batch_single_item(client: EmbeddingClient):
    """encode_batch() with a single item should return shape (1, 1024)."""
    embs = client.encode_batch(["single"])
    assert embs.shape == (1, 1024)


# ---------------------------------------------------------------------------
# Test: cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_text(client: EmbeddingClient):
    """cosine_similarity of a text with itself should be ≈ 1.0."""
    text = "The quick brown fox jumps over the lazy dog."
    emb = client.encode(text)
    sim = EmbeddingClient.cosine_similarity(emb, emb)
    assert abs(sim - 1.0) < 1e-5, f"Expected ~1.0, got {sim}"


def test_cosine_similarity_similar_texts(client: EmbeddingClient):
    """Semantically similar sentences should have high cosine similarity (> 0.8)."""
    text1 = "The cat sat on the mat."
    text2 = "A cat was sitting on a mat."
    emb1 = client.encode(text1)
    emb2 = client.encode(text2)
    sim = EmbeddingClient.cosine_similarity(emb1, emb2)
    assert sim > 0.8, f"Similar sentences should have similarity > 0.8, got {sim}"


def test_cosine_similarity_different_texts(client: EmbeddingClient):
    """Semantically unrelated sentences should have lower cosine similarity (< 0.9)."""
    text1 = "The capital of France is Paris."
    text2 = "Quantum mechanics describes subatomic particle behaviour."
    emb1 = client.encode(text1)
    emb2 = client.encode(text2)
    sim = EmbeddingClient.cosine_similarity(emb1, emb2)
    # These topics are unrelated; similarity should be noticeably below 1.0
    assert sim < 0.9, f"Different texts should have similarity < 0.9, got {sim}"


def test_cosine_similarity_opposite_texts(client: EmbeddingClient):
    """Embeddings of antonyms should be less similar than same text (< 1.0)."""
    text1 = "I love sunny weather."
    text2 = "I hate rainy weather."
    emb1 = client.encode(text1)
    emb2 = client.encode(text2)
    sim = EmbeddingClient.cosine_similarity(emb1, emb2)
    # Antonyms on the same topic tend to be moderately similar; the key is it's < 1.0
    assert sim < 1.0, f"Antonym texts should not have similarity == 1.0, got {sim}"


# ---------------------------------------------------------------------------
# Test: encode and encode_batch give consistent results
# ---------------------------------------------------------------------------


def test_encode_consistent_with_encode_batch(client: EmbeddingClient):
    """encode() and encode_batch() should return the same vector for the same text."""
    text = "Consistency check for single vs batch."
    single = client.encode(text)
    batch = client.encode_batch([text])[0]
    np.testing.assert_allclose(
        single, batch, atol=1e-5, err_msg="encode and encode_batch results differ"
    )


# ---------------------------------------------------------------------------
# Test: unload() / reload()
# ---------------------------------------------------------------------------


def test_unload_resets_is_loaded(client: EmbeddingClient):
    """unload() should reset is_loaded to False."""
    _ = client.encode("trigger load")
    assert client.is_loaded
    client.unload()
    assert not client.is_loaded


def test_reload_after_unload(client: EmbeddingClient):
    """reload() should re-load the model after an unload."""
    _ = client.encode("trigger load")
    assert client.is_loaded
    client.unload()
    assert not client.is_loaded
    client.reload()
    assert client.is_loaded


def test_reload_produces_same_embedding(client: EmbeddingClient):
    """Embedding values should be identical before and after reload."""
    text = "Test text for reload consistency."
    emb_before = client.encode(text)
    client.reload()
    emb_after = client.encode(text)
    np.testing.assert_allclose(
        emb_before, emb_after, atol=1e-5, err_msg="Embedding changed after reload"
    )

