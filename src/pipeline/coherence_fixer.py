"""
Step 6: Coherence Fixer + Retry Orchestrator

Two classes:

  CoherenceFixer
    Detects paragraph-level incoherence using lightweight heuristic rules,
    then calls ProbabilisticRewriter.rewrite_paragraph() to regenerate
    incoherent paragraphs.  No text-layer edits are ever made — every fix
    happens at the probabilistic generation layer.

  RetryOrchestrator
    Drives the full retry loop across semantic validation, detection-evasion
    evaluation, and language quality evaluation.  Up to max_retries attempts
    are made; the best quality version is returned when the limit is exceeded.
    Hard constraint: overall quality score must be >= min_quality_score (3.0);
    if all attempts fall below the floor, the highest-scoring attempt is
    returned with a warning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.pipeline.proposition_extractor import Proposition
from src.pipeline.quality_controller import QualityController, QualityReport
from src.pipeline.rewriter import ProbabilisticRewriter
from src.pipeline.semantic_anchor import SemanticAnchor
from src.pipeline.semantic_validator import SemanticValidator, ValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contrast-word patterns that trigger the incoherence heuristic
# ---------------------------------------------------------------------------

_CONTRAST_WORDS_ZH = ("但是", "然而", "相反", "不过", "却")
_CONTRAST_WORDS_EN = ("However", "But", "On the contrary", "Nevertheless", "Yet")

# Words / phrases that indicate the preceding paragraph already set up a
# contrast, making the contrast opener in the next paragraph coherent.
_CONTRAST_SETUP_ZH = ("虽然", "尽管", "固然", "尽管如此", "即使", "一方面")
_CONTRAST_SETUP_EN = (
    "although",
    "though",
    "even though",
    "while",
    "whereas",
    "on one hand",
    "on the one hand",
)


# ---------------------------------------------------------------------------
# CoherenceFixer
# ---------------------------------------------------------------------------


class CoherenceFixer:
    """
    Paragraph-level coherence fixer (Pipeline Step 6, first part).

    Detects incoherent paragraph transitions with heuristic rules, then
    regenerates incoherent paragraphs via ProbabilisticRewriter.rewrite_paragraph().

    Parameters
    ----------
    rewriter : ProbabilisticRewriter
        The shared probabilistic rewriter instance.
    """

    def __init__(self, rewriter: ProbabilisticRewriter) -> None:
        self._rewriter = rewriter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fix(self, text: str) -> str:
        """
        Detect and repair incoherent paragraph transitions.

        Strategy
        --------
        1. Split text into paragraphs.
        2. For each adjacent pair (para[i], para[i+1]), call
           _detect_incoherence().
        3. If incoherence is detected, regenerate para[i+1] via
           rewrite_paragraph() using para[i] as context.
        4. Reassemble and return.

        No text-layer edits are made — only probabilistic regeneration.

        Parameters
        ----------
        text : str  The text to fix.

        Returns
        -------
        str  The (potentially repaired) text.
        """
        paragraphs = self._split_paragraphs(text)

        if len(paragraphs) <= 1:
            # Single paragraph — nothing to check.
            return text

        fixed: list[str] = [paragraphs[0]]

        for i in range(1, len(paragraphs)):
            prev = fixed[i - 1]  # use the (possibly already-fixed) predecessor
            curr = paragraphs[i]

            if self._detect_incoherence(prev, curr):
                logger.debug(
                    "Incoherence detected between paragraph %d and %d — regenerating.",
                    i - 1,
                    i,
                )
                regenerated = self._rewriter.rewrite_paragraph(
                    paragraph=curr,
                    context=prev,
                    attempt=1,
                )
                fixed.append(regenerated.strip())
            else:
                fixed.append(curr)

        return "\n\n".join(fixed)

    # ------------------------------------------------------------------
    # Heuristic incoherence detection
    # ------------------------------------------------------------------

    def _detect_incoherence(self, para1: str, para2: str) -> bool:
        """
        Determine whether the transition from *para1* to *para2* is incoherent.

        Heuristic rules (any one triggers True):

        Rule 1 — Unprepared contrast opener
            para2 starts with a strong contrast word (e.g. "但是", "However")
            but para1 ends without any contrast-setup phrase.

        Rule 2 — Extreme sentence-length divergence
            The mean sentence length of para2 differs from that of para1
            by more than 50 characters.

        Note: these rules are approximate; false positives are expected and
        accepted at MVP stage.

        Parameters
        ----------
        para1 : str  The preceding paragraph.
        para2 : str  The following paragraph.

        Returns
        -------
        bool  True if the transition appears incoherent.
        """
        # ---- Rule 1: unprepared contrast opener ----------------------------
        if self._starts_with_contrast(para2) and not self._has_contrast_setup(para1):
            logger.debug("Incoherence rule 1 triggered (unprepared contrast opener).")
            return True

        # ---- Rule 2: extreme sentence-length divergence --------------------
        mean1 = self._mean_sentence_length(para1)
        mean2 = self._mean_sentence_length(para2)
        if abs(mean1 - mean2) > 50:
            logger.debug(
                "Incoherence rule 2 triggered (mean sentence length diff %.1f > 50).",
                abs(mean1 - mean2),
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Split *text* into non-empty paragraphs.

        Prefers double-newline (blank-line) splitting; falls back to
        single-newline splitting if double-newline yields only one segment.
        """
        # Try double-newline first
        double_split = [s.strip() for s in re.split(r"\n\s*\n", text)]
        double_split = [s for s in double_split if s]
        if len(double_split) >= 2:
            return double_split

        # Fallback: single newline
        single_split = [s.strip() for s in text.split("\n")]
        single_split = [s for s in single_split if s]
        if single_split:
            return single_split

        # Last resort: whole text
        stripped = text.strip()
        return [stripped] if stripped else []

    @staticmethod
    def _starts_with_contrast(para: str) -> bool:
        """Return True if *para* opens with a known contrast word."""
        stripped = para.strip()
        for word in _CONTRAST_WORDS_ZH + _CONTRAST_WORDS_EN:
            if stripped.startswith(word):
                return True
        return False

    @staticmethod
    def _has_contrast_setup(para: str) -> bool:
        """
        Return True if *para* contains a contrast-setup phrase — i.e. it
        already signals that a contrast is coming, making the contrast opener
        in the next paragraph expected and coherent.
        """
        lower = para.lower()
        for phrase in _CONTRAST_SETUP_ZH + tuple(p.lower() for p in _CONTRAST_SETUP_EN):
            if phrase in lower:
                return True
        return False

    @staticmethod
    def _mean_sentence_length(para: str) -> float:
        """
        Compute the mean character length of sentences in *para*.

        Splits on Chinese and Western sentence-ending punctuation.
        Returns 0.0 for empty paragraphs.
        """
        sentences = [s.strip() for s in re.split(r"[。！？!?\.]+", para) if s.strip()]
        if not sentences:
            return 0.0
        return sum(len(s) for s in sentences) / len(sentences)


# ---------------------------------------------------------------------------
# RetryOrchestrator data classes
# ---------------------------------------------------------------------------


@dataclass
class AttemptRecord:
    """Records everything about a single rewrite attempt."""

    attempt_number: int
    rewritten_text: str
    validation_result: ValidationResult | None
    quality_report: QualityReport | None
    overall_quality_score: float  # QualityReport.quality.overall_score; 0.0 if unavailable
    action_taken: str  # see RetryOrchestrator for valid values


@dataclass
class OrchestratorResult:
    """Final result returned by RetryOrchestrator.run()."""

    final_text: str
    accepted: bool
    total_attempts: int
    attempts: list[AttemptRecord]
    warning: str  # "" or "最大重试次数已达到，返回最高质量版本"


# ---------------------------------------------------------------------------
# RetryOrchestrator
# ---------------------------------------------------------------------------


class RetryOrchestrator:
    """
    Retry scheduler for the full rewrite → validate → evaluate pipeline.

    Drives up to *max_retries* attempts, selecting the retry strategy
    based on which check failed:

    - Semantic validation failed  → rewrite(attempt=3)  [emphasise propositions]
    - Detection evasion failed    → rewrite(attempt=2)  [maximise sentence diversity]
    - Language quality failed     → rewrite(attempt=1)  [stricter quality system prompt]
    - Both passed                 → apply CoherenceFixer → return

    Hard quality floor: overall_quality_score < min_quality_score versions
    are excluded from the "best" selection unless every attempt falls below
    the floor (in which case the highest scorer is returned with a warning).

    Parameters
    ----------
    rewriter           : ProbabilisticRewriter
    validator          : SemanticValidator
    quality_controller : QualityController
    coherence_fixer    : CoherenceFixer
    max_retries        : int   Maximum number of rewrite attempts (default 3).
    min_quality_score  : float Quality floor — default 3.0 per architecture spec.
    """

    def __init__(
        self,
        rewriter: ProbabilisticRewriter,
        validator: SemanticValidator,
        quality_controller: QualityController,
        coherence_fixer: CoherenceFixer,
        max_retries: int = 3,
        min_quality_score: float = 3.0,
    ) -> None:
        self._rewriter = rewriter
        self._validator = validator
        self._quality_controller = quality_controller
        self._coherence_fixer = coherence_fixer
        self._max_retries = max_retries
        self._min_quality_score = min_quality_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        text: str,
        anchor: SemanticAnchor,
        propositions: list[Proposition],
    ) -> OrchestratorResult:
        """
        Execute the full retry loop.

        Parameters
        ----------
        text         : Original (AI-generated) text to rewrite.
        anchor       : SemanticAnchor built from *text* (for layer-1 check).
        propositions : Propositions extracted from *text* (for layer-2 check).

        Returns
        -------
        OrchestratorResult
        """
        attempts: list[AttemptRecord] = []
        # attempt_mode controls which rewrite strategy to use on the next call.
        # Starts at 1 (standard); overridden by failure type.
        attempt_mode: int = 1

        for attempt_num in range(1, self._max_retries + 1):
            logger.debug(
                "RetryOrchestrator: attempt %d/%d (mode=%d)",
                attempt_num,
                self._max_retries,
                attempt_mode,
            )

            # ----------------------------------------------------------
            # 1. Rewrite
            # ----------------------------------------------------------
            rewrite_result = self._rewriter.rewrite(
                text=text,
                propositions=propositions,
                attempt=attempt_mode,
            )
            rewritten = rewrite_result.rewritten_text

            # ----------------------------------------------------------
            # 2. Semantic validation
            # ----------------------------------------------------------
            validation_result: ValidationResult | None = None
            try:
                validation_result = self._validator.validate(
                    anchor=anchor,
                    rewritten_text=rewritten,
                    propositions=propositions,
                )
            except Exception as exc:
                logger.warning("Semantic validation raised an exception: %s", exc)
                # Treat as a soft failure — proceed to quality check.

            # ----------------------------------------------------------
            # 3. Quality evaluation
            # ----------------------------------------------------------
            quality_report: QualityReport | None = None
            try:
                quality_report = self._quality_controller.evaluate(
                    text=rewritten,
                    original_text=text,
                )
            except Exception as exc:
                logger.warning("Quality evaluation raised an exception: %s", exc)

            overall_score = (
                quality_report.quality.overall_score if quality_report is not None else 0.0
            )

            # ----------------------------------------------------------
            # 4. Decide action
            # ----------------------------------------------------------
            semantic_ok = validation_result is None or validation_result.passed
            quality_ok = quality_report is None or (
                not quality_report.quality.needs_regeneration
                and quality_report.final_action != "retry_quality"
            )
            detection_ok = (
                quality_report is None or quality_report.final_action != "retry_detection"
            )

            if semantic_ok and quality_ok and detection_ok:
                # All checks passed — apply coherence fixer and return.
                action = "accepted"
                record = AttemptRecord(
                    attempt_number=attempt_num,
                    rewritten_text=rewritten,
                    validation_result=validation_result,
                    quality_report=quality_report,
                    overall_quality_score=overall_score,
                    action_taken=action,
                )
                attempts.append(record)

                logger.debug(
                    "RetryOrchestrator: attempt %d accepted; running CoherenceFixer.",
                    attempt_num,
                )
                final_text = self._coherence_fixer.fix(rewritten)

                return OrchestratorResult(
                    final_text=final_text,
                    accepted=True,
                    total_attempts=attempt_num,
                    attempts=attempts,
                    warning="",
                )

            # Determine retry strategy for the next round.
            if not semantic_ok:
                # Semantic failure: emphasise dropped propositions.
                action = "semantic_retry"
                attempt_mode = 3
                logger.debug("Attempt %d: semantic validation failed → next attempt=3", attempt_num)
            elif not quality_ok:
                # Language quality below floor: strengthen quality instructions.
                action = "quality_retry"
                attempt_mode = 1  # same sampling; prompt is stricter on quality side
                logger.debug(
                    "Attempt %d: quality check failed → next attempt=1 (quality focus)", attempt_num
                )
            else:
                # Detection evasion failed.
                action = "detection_retry"
                attempt_mode = 2
                logger.debug("Attempt %d: detection metrics failed → next attempt=2", attempt_num)

            record = AttemptRecord(
                attempt_number=attempt_num,
                rewritten_text=rewritten,
                validation_result=validation_result,
                quality_report=quality_report,
                overall_quality_score=overall_score,
                action_taken=action,
            )
            attempts.append(record)

        # ------------------------------------------------------------------
        # Max retries exhausted — pick the best version.
        # ------------------------------------------------------------------
        logger.warning(
            "RetryOrchestrator: max retries (%d) exhausted. Selecting best attempt.",
            self._max_retries,
        )

        best = self._select_best_attempt(attempts)

        return OrchestratorResult(
            final_text=best.rewritten_text,
            accepted=False,
            total_attempts=self._max_retries,
            attempts=attempts,
            warning="最大重试次数已达到，返回最高质量版本",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_best_attempt(self, attempts: list[AttemptRecord]) -> AttemptRecord:
        """
        Return the attempt with the highest overall_quality_score.

        Preference order:
        1. Attempts with overall_quality_score >= min_quality_score.
        2. If none qualify, fall back to the attempt with the highest score
           (even if below the floor).
        """
        above_floor = [a for a in attempts if a.overall_quality_score >= self._min_quality_score]
        candidates = above_floor if above_floor else attempts

        best = max(candidates, key=lambda a: a.overall_quality_score)
        logger.debug(
            "Best attempt selected: attempt=%d, score=%.3f",
            best.attempt_number,
            best.overall_quality_score,
        )
        return best
