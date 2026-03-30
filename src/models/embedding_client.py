"""
Local embedding client using BAAI/bge-m3 via sentence-transformers.

Design notes
------------
- The SentenceTransformer model is loaded lazily on first use to avoid
  slowing down imports and to allow unit tests that don't need the model
  to run without downloading it.
- All computation is forced to CPU (per architecture spec: BGE-M3 always
  runs on CPU, not on the GPU reserved for the generation model).
- Embeddings are L2-normalised so that cosine similarity reduces to a
  plain dot product (faster, numerically stable).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Wrapper around a local sentence-transformer model.

    Parameters
    ----------
    config : EmbeddingConfig
        Configuration object. If None, defaults are used (BAAI/bge-m3, CPU).
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.cfg = config or EmbeddingConfig()
        self._model = None  # lazy-loaded on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single string into a dense embedding vector.

        Parameters
        ----------
        text : str  Input text.

        Returns
        -------
        np.ndarray  Shape (embedding_dim,), dtype float32.
        """
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into a 2-D embedding matrix.

        Parameters
        ----------
        texts : list[str]  Input texts.

        Returns
        -------
        np.ndarray  Shape (len(texts), embedding_dim), dtype float32.
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Because embeddings are L2-normalised by default, this is equivalent
        to the dot product — but we compute it explicitly so that the method
        is correct even when called with non-normalised vectors.

        Parameters
        ----------
        emb1, emb2 : np.ndarray  1-D embedding vectors of the same dimension.

        Returns
        -------
        float  Cosine similarity in [-1, 1].
        """
        emb1 = emb1.flatten().astype(np.float64)
        emb2 = emb2.flatten().astype(np.float64)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying model has been loaded into memory."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Return the SentenceTransformer model, loading it on first call."""
        if self._model is None:
            logger.info("Loading embedding model %s on %s …", self.cfg.model_name, self.cfg.device)
            # Import here so that the module can be imported without
            # sentence-transformers being installed (e.g. in CI environments
            # that only need the LLM client).
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(
                self.cfg.model_name,
                device=self.cfg.device,
            )
            logger.info("Embedding model loaded.")
        return self._model
