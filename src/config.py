"""
ScholAI configuration management.
Reads settings from environment variables (with .env file support via python-dotenv).
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file if present (no-op if it doesn't exist)
load_dotenv()


@dataclass
class LLMConfig:
    """DeepSeek LLM API configuration."""

    api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    base_url: str = "https://api.deepseek.com"

    # Primary chat model
    model: str = "deepseek-chat"

    # Sampling parameters for probabilistic-layer rewriting
    # These approximate Mirostat v2 behaviour since DeepSeek API doesn't expose
    # Mirostat natively.
    temperature: float = 0.9
    top_p: float = 0.85
    frequency_penalty: float = 0.3   # suppresses repetitive AI patterns
    presence_penalty: float = 0.0

    # Generation limits
    max_tokens: int = 4096

    # Retry / backoff
    max_retries: int = 3
    retry_min_wait: float = 1.0   # seconds
    retry_max_wait: float = 8.0   # seconds

    # Request timeout (seconds)
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. "
                "Export it as an environment variable or add it to your .env file."
            )


@dataclass
class EmbeddingConfig:
    """Local sentence-transformer embedding configuration."""

    model_name: str = "BAAI/bge-m3"
    # Force CPU — BGE-M3 is always on CPU per architecture spec
    device: str = "cpu"
    # Batch size for encode_batch
    batch_size: int = 32
    # Normalise embeddings to unit length (required for cosine similarity via dot product)
    normalize_embeddings: bool = True


@dataclass
class AppConfig:
    """Top-level configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


def get_config() -> AppConfig:
    """Return a fully-initialised AppConfig, reading from the environment."""
    return AppConfig()
