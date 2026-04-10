"""
ScholAI model clients: LLM and embedding.
"""

from src.models.embedding_client import EmbeddingClient
from src.models.llm_client import LLMClient

__all__ = ["LLMClient", "EmbeddingClient"]
