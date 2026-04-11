"""
ScholAI test suite.

Shared fixtures and helpers are defined in conftest.py.
This module re-exports the get_api_key helper for convenience.

Usage
-----
    from tests import get_api_key
"""

from tests.conftest import get_api_key

__all__ = ["get_api_key"]
