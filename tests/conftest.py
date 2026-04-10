"""
Pytest configuration for ScholAI tests.

Loads DEEPSEEK_API_KEY from the project .env file (if present) so that
individual test modules do not need to hard-code credentials.
pytest-dotenv is loaded automatically when this file is present.

Usage in test modules
--------------------
    from tests.conftest import DEEPSEEK_API_KEY   # constant — safe to import
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

Or simply rely on the autouse fixtures defined below.
"""

from __future__ import annotations

import os
from pathlib import Path

# Resolve project root (tests/) → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env so that os.environ["DEEPSEEK_API_KEY"] is populated before
# any src.config import runs (src.config calls load_dotenv() on import).
_DOTENV_PATH = _REPO_ROOT / ".env"
if _DOTENV_PATH.exists():
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=_DOTENV_PATH, override=True)

# Validate that the key is available
_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not _DEEPSEEK_API_KEY:
    raise RuntimeError(
        "DEEPSEEK_API_KEY is not set in the environment and not found in "
        f"{_DOTENV_PATH}. Copy .env.example to .env and fill in your key."
    )


def get_api_key() -> str:
    """Return the DeepSeek API key loaded from .env."""
    return _DEEPSEEK_API_KEY
