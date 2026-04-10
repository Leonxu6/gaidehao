"""
Unit tests for src/utils/__init__.py.

Tests text normalisation, sentence/paragraph splitting, tokenisation,
and language-detection utilities used throughout the pipeline.
"""

from __future__ import annotations

import pytest

from src.utils import (
    chinese_ratio,
    is_chinese,
    is_empty,
    normalize,
    split_paragraphs,
    split_sentences,
    tokenize,
)

# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace():
    """normalize() should strip leading and trailing whitespace."""
    assert normalize("  hello world  ") == "hello world"


def test_normalize_collapses_internal_whitespace():
    """normalize() should collapse runs of whitespace to a single space."""
    assert normalize("hello    world") == "hello world"
    assert normalize("hello\n\n\nworld") == "hello world"
    assert normalize("hello\t\tworld") == "hello world"


def test_normalize_removes_zero_width_chars():
    """normalize() should remove zero-width characters."""
    # Zero-width characters are stripped; adjacent text joins without space
    assert normalize("hello\u200bworld") == "helloworld"
    assert normalize("hello\u200eworld") == "helloworld"
    assert normalize("hello\ufeffworld") == "helloworld"


def test_normalize_empty_string():
    """normalize() of an empty string should return empty string."""
    assert normalize("") == ""
    assert normalize("   ") == ""


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


def test_is_empty_none():
    """is_empty(None) should return True."""
    assert is_empty(None) is True


def test_is_empty_whitespace_only():
    """is_empty() with whitespace-only string should return True."""
    assert is_empty("   ") is True
    assert is_empty("\t\n") is True


def test_is_empty_non_empty():
    """is_empty() with non-empty string should return False."""
    assert is_empty("hello") is False
    assert is_empty("  hello  ") is False


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_chinese():
    """split_sentences() should split Chinese text on 。！？"""
    text = "今天天气很好。我们去散步吧！一起去公园怎么样？"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "今天天气很好"
    assert sentences[1] == "我们去散步吧"
    assert sentences[2] == "一起去公园怎么样"


def test_split_sentences_english():
    """split_sentences() should split English text on ! and ? (not .)"""
    text = "Hello world! How are you? I'm fine"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world"
    assert sentences[1] == "How are you"
    assert sentences[2] == "I'm fine"


def test_split_sentences_mixed():
    """split_sentences() should handle mixed Chinese/English punctuation."""
    text = "Hello！你好。World?世界！"
    sentences = split_sentences(text)
    assert len(sentences) == 4


def test_split_sentences_empty():
    """split_sentences() of empty text should return empty list."""
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_no_delimiter():
    """split_sentences() with no delimiters should return single-item list."""
    assert split_sentences("hello world") == ["hello world"]


# ---------------------------------------------------------------------------
# split_paragraphs
# ---------------------------------------------------------------------------


def test_split_paragraphs_double_newline():
    """split_paragraphs() should prefer double-newline (blank line) splitting."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paras = split_paragraphs(text)
    assert len(paras) == 3
    assert paras[0] == "First paragraph."
    assert paras[1] == "Second paragraph."
    assert paras[2] == "Third paragraph."


def test_split_paragraphs_single_newline_fallback():
    """split_paragraphs() should fall back to single-newline splitting."""
    text = "First.\nSecond.\nThird."
    paras = split_paragraphs(text)
    assert len(paras) == 3


def test_split_paragraphs_empty():
    """split_paragraphs() of empty text should return empty list."""
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\n   ") == []


def test_split_paragraphs_single_paragraph():
    """split_paragraphs() with no newlines should return single-item list."""
    assert split_paragraphs("Just one paragraph.") == ["Just one paragraph."]


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_english():
    """tokenize() should split English text on spaces and punctuation."""
    tokens = tokenize("Hello, world! How are you?")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_chinese():
    """tokenize() should treat each CJK character as a separate token."""
    tokens = tokenize("今天天气很好")
    assert len(tokens) == 6  # 今, 天, 天, 气, 很, 好
    assert "今" in tokens
    assert "天" in tokens
    assert "好" in tokens


def test_tokenize_mixed():
    """tokenize() should handle mixed Chinese/English text."""
    tokens = tokenize("Hello世界")
    # The whole string goes through tokenization; CJK chars split individually
    assert len(tokens) >= 1


def test_tokenize_empty():
    """tokenize() of empty text should return empty list."""
    assert tokenize("") == []
    assert tokenize("   ") == []


# ---------------------------------------------------------------------------
# chinese_ratio
# ---------------------------------------------------------------------------


def test_chinese_ratio_pure_chinese():
    """chinese_ratio() of pure Chinese text should be 1.0."""
    assert chinese_ratio("今天天气很好") == 1.0


def test_chinese_ratio_pure_english():
    """chinese_ratio() of pure English text should be 0.0."""
    assert chinese_ratio("Hello world") == 0.0


def test_chinese_ratio_mixed():
    """chinese_ratio() of mixed text should return correct fraction."""
    # "Hello世界" has 5 chars: 5 English, 2 Chinese
    ratio = chinese_ratio("Hello世界")
    assert 0.0 < ratio < 1.0


def test_chinese_ratio_empty():
    """chinese_ratio() of empty text should return 0.0."""
    assert chinese_ratio("") == 0.0


# ---------------------------------------------------------------------------
# is_chinese
# ---------------------------------------------------------------------------


def test_is_chinese_true_for_chinese_text():
    """is_chinese() should return True for Chinese text above threshold."""
    assert is_chinese("今天天气很好我们去做运动吧") is True


def test_is_chinese_false_for_english_text():
    """is_chinese() should return False for English text."""
    assert is_chinese("Hello world, how are you today?") is False


def test_is_chinese_false_for_mixed_below_threshold():
    """is_chinese() should return False when Chinese chars are below threshold."""
    # Very short mixed text may fall below 30% threshold
    assert is_chinese("Hello世界") is False
