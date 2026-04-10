"""
Step 2: Semantic Anchor Embedding

Builds a SemanticAnchor from raw text by embedding the full document and
each paragraph with BAAI/bge-m3 (via EmbeddingClient).  The anchor is
consumed by Step 4 (first-layer coarse semantic validation).

Design notes
------------
- Full-document embedding captures overall topic/stance.
- Paragraph-level embeddings allow per-chunk drift detection.
- Paragraph splitting prefers double-newlines; single newlines are used
  as a fallback so that single-newline-delimited texts still get chunked.
- All cosine-similarity computation is delegated to EmbeddingClient so
  that normalisation behaviour is consistent across the pipeline.
- Default coarse-screening threshold is 0.80, matching the architecture
  spec (Section 四, 第一层).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from src.models.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SemanticAnchor:
    """Holds the embedding artefacts for one original document."""

    embedding: np.ndarray  # Full-document embedding, shape (D,)
    original_text: str  # Verbatim original text
    chunk_embeddings: list[np.ndarray]  # Per-paragraph embeddings, each shape (D,)
    chunks: list[str]  # Corresponding paragraph texts
    similarity_threshold: float = 0.80  # Coarse-screening pass/fail boundary

    def __repr__(self) -> str:
        return (
            f"SemanticAnchor("
            f"embedding_dim={self.embedding.shape[0]}, "
            f"n_chunks={len(self.chunks)}, "
            f"text_len={len(self.original_text)}, "
            f"threshold={self.similarity_threshold}"
            f")"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class SemanticAnchorBuilder:
    """
    Pipeline wrapper that converts raw text into a SemanticAnchor.

    Parameters
    ----------
    embedding_client : EmbeddingClient
        An already-instantiated client wrapping BAAI/bge-m3.
    similarity_threshold : float
        Minimum cosine similarity required for the coarse semantic check.
        Default 0.80 per architecture spec.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        similarity_threshold: float = 0.80,
    ) -> None:
        self._client = embedding_client
        self._similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, text: str) -> SemanticAnchor:
        """
        Embed *text* at both full-document and paragraph levels.

        Steps
        -----
        1. Encode the full document as a single embedding.
        2. Split into paragraphs (double-newline preferred, single-newline
           as fallback), filter empty segments.
        3. Encode every paragraph individually.
        4. Return a SemanticAnchor containing all artefacts.

        Parameters
        ----------
        text : str
            The original (AI-generated) document to anchor.

        Returns
        -------
        SemanticAnchor
        """
        # 1. Full-document embedding
        logger.debug("Encoding full document (length=%d chars).", len(text))
        full_embedding: np.ndarray = self._client.encode(text)

        # 2. Paragraph splitting
        chunks = self._split_paragraphs(text)
        logger.debug("Extracted %d paragraph chunk(s).", len(chunks))

        # 3. Paragraph-level embeddings (batch call for efficiency)
        if chunks:
            chunk_matrix = self._client.encode_batch(chunks)
            chunk_embeddings: list[np.ndarray] = [chunk_matrix[i] for i in range(len(chunks))]
        else:
            chunk_embeddings = []

        return SemanticAnchor(
            embedding=full_embedding,
            original_text=text,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            similarity_threshold=self._similarity_threshold,
        )

    def check_similarity(self, anchor: SemanticAnchor, rewritten_text: str) -> dict:
        """
        Coarse full-document semantic check (Step 4, Layer 1).

        Embeds *rewritten_text* and computes cosine similarity against the
        anchor's full-document embedding.

        Parameters
        ----------
        anchor : SemanticAnchor
            Anchor built from the original text.
        rewritten_text : str
            Candidate rewritten document.

        Returns
        -------
        dict with keys:
            similarity : float   Cosine similarity in [-1, 1].
            passed     : bool    True if similarity >= anchor.similarity_threshold.
            reason     : str     Human-readable explanation.
        """
        rewritten_emb = self._client.encode(rewritten_text)
        similarity = EmbeddingClient.cosine_similarity(anchor.embedding, rewritten_emb)
        passed = similarity >= anchor.similarity_threshold

        if passed:
            reason = (
                f"Semantic similarity {similarity:.4f} meets threshold "
                f"{anchor.similarity_threshold:.2f}."
            )
        else:
            reason = (
                f"Semantic similarity {similarity:.4f} is below threshold "
                f"{anchor.similarity_threshold:.2f}; rewrite rejected."
            )

        logger.debug("check_similarity: similarity=%.4f, passed=%s", similarity, passed)
        return {"similarity": similarity, "passed": passed, "reason": reason}

    def check_chunk_similarity(
        self,
        anchor: SemanticAnchor,
        rewritten_text: str,
    ) -> list[dict]:
        """
        Fine-grained per-paragraph semantic check.

        For each paragraph in the original anchor, find the rewritten
        paragraph with the highest cosine similarity and report whether it
        passes the threshold.

        All similarities are computed in a single matrix multiplication
        (vectorised) rather than nested loops — O(n*m) but with one matmul
        instead of n*m individual dot products.

        Parameters
        ----------
        anchor : SemanticAnchor
            Anchor built from the original text.
        rewritten_text : str
            Candidate rewritten document.

        Returns
        -------
        list[dict], one entry per original chunk, each with keys:
            chunk_idx  : int    Index into anchor.chunks.
            original   : str    Original paragraph text.
            similarity : float  Highest similarity found in rewritten paragraphs.
            passed     : bool   True if similarity >= anchor.similarity_threshold.
        """
        if not anchor.chunk_embeddings:
            return []

        rewritten_chunks = self._split_paragraphs(rewritten_text)
        if not rewritten_chunks:
            # Treat the whole rewritten text as one chunk
            rewritten_chunks = [rewritten_text]

        rewritten_matrix = self._client.encode_batch(rewritten_chunks)

        # Vectorised similarity: (n, D) @ (D, m).T = (n, m)  — one matmul
        # instead of the previous O(n*m) nested loop.
        orig_matrix = np.stack(anchor.chunk_embeddings)  # (n, D)
        sim_matrix = orig_matrix @ rewritten_matrix.T  # (n, m)

        # Best similarity per original chunk (row-wise max)
        best_sims = sim_matrix.max(axis=1)  # (n,)

        results: list[dict] = [
            {
                "chunk_idx": idx,
                "original": chunk,
                "similarity": float(best_sims[idx]),
                "passed": bool(best_sims[idx] >= anchor.similarity_threshold),
            }
            for idx, chunk in enumerate(anchor.chunks)
        ]

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """
        Split *text* into non-empty paragraphs.

        Strategy
        --------
        1. Try splitting on one-or-more blank lines (double/triple newlines).
           If this yields >= 2 non-empty segments, use it.
        2. Otherwise fall back to single-newline splitting.
        3. Strip each segment; discard empty strings.
        """
        # Attempt double-newline split first
        double_nl_chunks = [s.strip() for s in re.split(r"\n\s*\n", text)]
        double_nl_chunks = [s for s in double_nl_chunks if s]

        if len(double_nl_chunks) >= 2:
            return double_nl_chunks

        # Fallback: single-newline split
        single_nl_chunks = [s.strip() for s in text.split("\n")]
        single_nl_chunks = [s for s in single_nl_chunks if s]

        if single_nl_chunks:
            return single_nl_chunks

        # Last resort: whole text as one chunk
        stripped = text.strip()
        return [stripped] if stripped else []
