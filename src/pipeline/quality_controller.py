"""
Step 5: Quality Controller — dual-track quality evaluation.

Track 1 (DetectionMetrics): statistical features that answer "does this text
look like it was written by a human?" — PPL estimate, burstiness, TTR.

Track 2 (QualityMetrics): linguistic quality that answers "is this text well
written?" — rule-based quick scan + G-Eval LLM scoring.

The two tracks are strictly separated: Track 1 says nothing about writing
quality; Track 2 says nothing about AI-detectability.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from src.models.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionMetrics:
    """Track 1: detection-evasion metrics — only answers 'does it look human?'"""
    ppl_estimate: float          # Unigram-entropy-based PPL approximation
    burstiness: float            # Sentence-length standard deviation
    ttr: float                   # Type-Token Ratio
    ppl_passed: bool             # 25 <= ppl_estimate <= 150
    burstiness_passed: bool      # burstiness >= 10
    ttr_passed: bool             # ttr >= 0.45
    overall_passed: bool         # all three must pass


@dataclass
class QualityMetrics:
    """Track 2: linguistic quality metrics — only answers 'is it well written?'"""
    rule_issues: list[str]       # Issues found by the rule-based quick scan
    rule_passed: bool
    faithfulness: float          # G-Eval dimension: 0-5
    coherence: float             # G-Eval dimension: 0-5
    fluency: float               # G-Eval dimension: 0-5
    readability: float           # G-Eval dimension: 0-5
    overall_score: float         # 0.35*f + 0.25*c + 0.25*fl + 0.15*r
    quality_passed: bool         # overall_score >= 3.5
    needs_regeneration: bool     # overall_score < 3.0


@dataclass
class QualityReport:
    detection: DetectionMetrics
    quality: QualityMetrics
    final_action: str  # "accept" | "retry_detection" | "retry_quality" | "retry_semantic"


# ---------------------------------------------------------------------------
# G-Eval prompt template
# ---------------------------------------------------------------------------

_GEVAL_PROMPT = """\
你是一个专业的中文写作质量评估员。请对以下文本进行评分。

【评分维度】
1. 忠实度（Faithfulness）：内容是否准确，有无凭空捏造
2. 连贯性（Coherence）：段落逻辑是否连贯，上下文是否衔接
3. 流畅性（Fluency）：语言是否自然，有无语法错误
4. 可读性（Readability）：结构是否清晰，用词是否恰当

【评分标准】5分:优秀 4分:良好 3分:一般 2分:较差 1分:很差

【待评文本】
{text}

请按以下JSON格式输出，不要有其他文字：
{{"faithfulness": 分数, "coherence": 分数, "fluency": 分数, "readability": 分数}}"""

_GEVAL_SYSTEM = (
    "你是一个专业的中文写作质量评估员。"
    "只输出要求的JSON，不要有任何额外文字。"
)


# ---------------------------------------------------------------------------
# QualityController
# ---------------------------------------------------------------------------

class QualityController:
    """
    Dual-track quality controller.

    Parameters
    ----------
    llm_client : LLMClient
        Shared LLM client used for G-Eval scoring (Track 2 only).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, text: str, original_text: str = "") -> QualityReport:
        """
        Run both tracks and return a QualityReport with a recommended action.

        Parameters
        ----------
        text          : The (rewritten) text to evaluate.
        original_text : The original input text (not used in current MVP but
                        kept for future faithfulness checks).
        """
        detection = self._compute_detection_metrics(text)
        quality = self._compute_quality_metrics(text)
        final_action = self._decide_action(detection, quality)
        return QualityReport(
            detection=detection,
            quality=quality,
            final_action=final_action,
        )

    # ------------------------------------------------------------------
    # Track 1 — Detection-evasion metrics
    # ------------------------------------------------------------------

    def _compute_detection_metrics(self, text: str) -> DetectionMetrics:
        ppl = self._estimate_ppl(text)
        burstiness = self._compute_burstiness(text)
        ttr = self._compute_ttr(text)

        # PPL upper bound is 150 (not 80) because our unigram-entropy estimate
        # runs much higher than true LM perplexity for English academic text —
        # a vocabulary-rich human paragraph routinely scores 100-140 by this
        # approximation while a genuine LM would score it ~30-60.
        ppl_passed = 25.0 <= ppl <= 150.0
        burstiness_passed = burstiness >= 10.0
        ttr_passed = ttr >= 0.45

        return DetectionMetrics(
            ppl_estimate=ppl,
            burstiness=burstiness,
            ttr=ttr,
            ppl_passed=ppl_passed,
            burstiness_passed=burstiness_passed,
            ttr_passed=ttr_passed,
            overall_passed=ppl_passed and burstiness_passed and ttr_passed,
        )

    def _estimate_ppl(self, text: str) -> float:
        """
        Approximate PPL via unigram entropy of the token frequency distribution.

        Formula: PPL ≈ 2^H  where H = -Σ p(w) log2 p(w)

        This is a rough but zero-dependency proxy.  Real PPL requires a language
        model; for MVP purposes this is sufficient to distinguish high-entropy
        (human-like) from low-entropy (AI-typical) texts.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 1.0
        counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
        return 2 ** entropy

    def _compute_burstiness(self, text: str) -> float:
        """
        Return the standard deviation of sentence lengths (in characters).

        Human writing has natural rhythm variation → high std dev.
        Uniform AI output → low std dev.
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.0
        lengths = [len(s) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        return math.sqrt(variance)

    def _compute_ttr(self, text: str) -> float:
        """
        Type-Token Ratio = unique_tokens / total_tokens.

        Higher TTR → richer vocabulary → more human-like.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using Chinese and Western punctuation.
        Returns non-empty sentences only.
        """
        parts = re.split(r"[。！？!?\.]+", text)
        return [p.strip() for p in parts if p.strip()]

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple character-level tokeniser for Chinese text with space/punctuation
        splitting for mixed content.

        Strategy:
        - Split on whitespace and common punctuation to get word-like tokens.
        - For any remaining CJK runs, treat each character as a token.
        This avoids a jieba dependency while remaining reasonable for Chinese.
        """
        # Split on spaces and common punctuation
        rough_tokens = re.split(
            r"[\s，。！？、；：\u201c\u201d\u2018\u2019（）【】《》〈〉\[\](),.!?;:\"'\-\u2014\u2026·]+",
            text,
        )
        tokens: list[str] = []
        for tok in rough_tokens:
            tok = tok.strip()
            if not tok:
                continue
            # Check if the token is purely CJK characters
            if re.fullmatch(r"[\u4e00-\u9fff\u3400-\u4dbf]+", tok):
                # Treat each CJK character as a separate token (unigram)
                tokens.extend(list(tok))
            else:
                tokens.append(tok.lower())
        return tokens

    # ------------------------------------------------------------------
    # Track 2 — Linguistic quality metrics
    # ------------------------------------------------------------------

    def _compute_quality_metrics(self, text: str) -> QualityMetrics:
        rule_passed, rule_issues = self._rule_check(text)
        geval = self._geval_score(text)

        faithfulness = geval.get("faithfulness", 3.0)
        coherence = geval.get("coherence", 3.0)
        fluency = geval.get("fluency", 3.0)
        readability = geval.get("readability", 3.0)

        overall = (
            0.35 * faithfulness
            + 0.25 * coherence
            + 0.25 * fluency
            + 0.15 * readability
        )

        return QualityMetrics(
            rule_issues=rule_issues,
            rule_passed=rule_passed,
            faithfulness=faithfulness,
            coherence=coherence,
            fluency=fluency,
            readability=readability,
            overall_score=overall,
            quality_passed=overall >= 3.5,
            needs_regeneration=overall < 3.0,
        )

    def _rule_check(self, text: str) -> tuple[bool, list[str]]:
        """
        Fast rule-based quality scan (millisecond-level, no model required).

        Checks:
        1. Duplicate consecutive sentences.
        2. Abnormal sentence length distribution (>30% too long or too short).
        3. Repeated punctuation marks (，， or 。。).
        """
        issues: list[str] = []
        sentences = re.split(r"[。！？!?\.]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        # 1. Duplicate consecutive sentences
        for i in range(1, len(sentences)):
            if sentences[i] == sentences[i - 1]:
                issues.append("重复句子")
                break

        # 2. Sentence length abnormality
        if sentences:
            long_sentences = [s for s in sentences if len(s) > 100]
            short_sentences = [s for s in sentences if len(s) < 8]
            ratio_long = len(long_sentences) / len(sentences)
            ratio_short = len(short_sentences) / len(sentences)
            if ratio_long > 0.3:
                issues.append("句子过长比例高")
            if ratio_short > 0.3:
                issues.append("句子过短比例高")

        # 3. Repeated punctuation
        if "，，" in text or "。。" in text:
            issues.append("标点重复")

        return len(issues) == 0, issues

    def _geval_score(self, text: str) -> dict:
        """
        Call the LLM once (temperature=0.0) with the G-Eval prompt and parse
        the returned JSON into a dict with keys:
        faithfulness, coherence, fluency, readability.

        Falls back to neutral scores (3.0) on parse failure.
        """
        prompt = self._build_geval_prompt(text)
        try:
            response = self._llm.generate(
                prompt=prompt,
                system_prompt=_GEVAL_SYSTEM,
                temperature=0.0,
                max_tokens=100,
            )
            return self._parse_geval_response(response)
        except Exception as exc:
            logger.warning("G-Eval scoring failed: %s — using fallback scores", exc)
            return {
                "faithfulness": 3.0,
                "coherence": 3.0,
                "fluency": 3.0,
                "readability": 3.0,
            }

    def _build_geval_prompt(self, text: str) -> str:
        """Insert text into the G-Eval prompt template."""
        return _GEVAL_PROMPT.format(text=text)

    def _parse_geval_response(self, response: str) -> dict:
        """
        Extract JSON from the LLM response.

        Tries strict json.loads first; falls back to a regex extraction in case
        the model wraps the JSON in markdown code fences or adds extra text.
        """
        # Try direct parse
        try:
            data = json.loads(response.strip())
            return self._validate_geval_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting the first JSON object via regex
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return self._validate_geval_dict(data)
            except (json.JSONDecodeError, ValueError):
                pass

        logger.warning("Could not parse G-Eval response: %r", response[:200])
        return {
            "faithfulness": 3.0,
            "coherence": 3.0,
            "fluency": 3.0,
            "readability": 3.0,
        }

    @staticmethod
    def _validate_geval_dict(data: dict) -> dict:
        """
        Validate that all four G-Eval keys are present and within [1, 5].
        Clamps values to the valid range and fills missing keys with 3.0.
        """
        keys = ("faithfulness", "coherence", "fluency", "readability")
        result = {}
        for k in keys:
            raw = data.get(k, 3.0)
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = 3.0
            result[k] = max(1.0, min(5.0, v))
        return result

    # ------------------------------------------------------------------
    # Final action decision
    # ------------------------------------------------------------------

    @staticmethod
    def _decide_action(
        detection: DetectionMetrics,
        quality: QualityMetrics,
    ) -> str:
        """
        Determine the recommended next action based on both tracks.

        Priority:
        1. Quality below regeneration threshold → retry quality (never sacrifice quality)
        2. Detection metrics failed → retry detection
        3. Otherwise → accept
        """
        if quality.needs_regeneration:
            return "retry_quality"
        if not detection.overall_passed:
            return "retry_detection"
        return "accept"
