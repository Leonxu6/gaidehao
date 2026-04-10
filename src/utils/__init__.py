"""
Shared text preprocessing utilities.

Provides consistent, language-aware text normalisation used across the
pipeline: PropositionExtractor, SemanticAnchorBuilder, QualityController,
and the rewriter.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """
    Collapse internal whitespace and strip leading/trailing whitespace.

    Unlike str.strip() this also removes zero-width characters and normalises
    Unicode whitespace to regular spaces, preventing subtle bugs in downstream
    tokenisation and embedding.
    """
    # Remove zero-width joiners and other invisible Unicode characters
    text = re.sub(r"[\u200b-\u200f\u2028-\u202f\ufeff]", "", text)
    # Collapse runs of whitespace to a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_empty(s: str) -> bool:
    """Return True if *s* is None or contains only whitespace."""
    return not s or not s.strip()


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_CHINESE_PUNCT = r"。！？!?"
_WESTERN_PUNCT = r".!?"


def split_sentences(text: str) -> list[str]:
    """
    Split *text* into individual sentences.

    Handles both Chinese and Western punctuation; consecutive delimiters
    are treated as a single break.  Returns non-empty sentences only.
    """
    parts = re.split(r"[。！？!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    """
    Split *text* into paragraphs.

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


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_CJK_RANGE = (
    r"\u4e00-\u9fff"  # CJK Unified Ideographs (core block)
    r"\u3400-\u4dbf"  # CJK Extension A
    r"\U00020000-\U0002a6df"  # CJK Extension B (surrogate pair range)
)

_CJK_TOKEN_RE = re.compile(f"[{_CJK_RANGE}]+")
_PUNCT_WHITESPACE_RE = re.compile(
    r"[\s，。！？、；：\u201c\u201d\u2018\u2019（）【】《》〈〉\[\](),.!?;:\"'\-\u2014\u2026·]+"
)


def tokenize(text: str) -> list[str]:
    """
    Character-level tokeniser for Chinese text with space/punctuation
    splitting for mixed content.

    Strategy
    --------
    - Split on whitespace and common punctuation to get word-like tokens.
    - For any remaining CJK runs, treat each character as a token.
    This avoids a jieba dependency while remaining reasonable for Chinese.
    """
    # Split on spaces and common punctuation
    rough_tokens = _PUNCT_WHITESPACE_RE.split(text)
    tokens: list[str] = []
    for tok in rough_tokens:
        tok = tok.strip()
        if not tok:
            continue
        # Check if the token is purely CJK characters
        if _CJK_TOKEN_RE.fullmatch(tok):
            # Treat each CJK character as a separate token (unigram)
            tokens.extend(list(tok))
        else:
            tokens.append(tok.lower())
    return tokens


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def chinese_ratio(text: str) -> float:
    """Return the fraction of characters in *text* that are CJK ideographs."""
    if not text:
        return 0.0
    cjk_count = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3400" <= ch <= "\u4dbf"
        or "\U00020000" <= ch <= "\U0002a6df"
    )
    return cjk_count / len(text)


# Default fraction of characters that must be CJK for the text to be
# treated as Chinese
CHINESE_CHAR_THRESHOLD = 0.30


def is_chinese(text: str) -> bool:
    """Return True if the dominant script in *text* is Chinese."""
    return chinese_ratio(text) > CHINESE_CHAR_THRESHOLD
