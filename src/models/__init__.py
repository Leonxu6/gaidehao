"""
ScholAI model clients: LLM and embedding.
"""

from src.models.llm_client import LLMClient
from src.models.embedding_client import EmbeddingClient

__all__ = ["LLMClient", "EmbeddingClient"]
