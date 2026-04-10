"""
Step 4: Semantic Validator — Three-Layer Semantic Verification

Validates that a rewritten text preserves the semantic content of the original
by running three sequential checks:

  Layer 1 (Coarse)   — Full-document cosine similarity via SemanticAnchorBuilder.
                        Pass threshold: ≥ 0.80.

  Layer 2 (Precise)  — Proposition-level NLI verification.
                        Uses cross-encoder/nli-deberta-v3-xsmall (~70 MB, CPU).
                        • entailment > 0.75  → proposition preserved  ✓
                        • contradiction > 0.70 → proposition contradicted → retry
                        • otherwise           → proposition missing

  Layer 3 (Auxiliary)— BERTScore word-coverage reference (F1 only).
                        Does NOT independently trigger a rewrite; failure in this
                        layer is purely informational.

Design notes
------------
- The NLI model is loaded lazily on first call to validate() / _layer2_check()
  and cached for the lifetime of the SemanticValidator instance.
- bert_score is an optional dependency; if it is not installed the layer-3 check
  gracefully returns None without affecting the main pipeline outcome.
- All exceptions in layer-3 are caught so that BERTScore failures never abort
  an otherwise-successful validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.pipeline.proposition_extractor import Proposition
from src.pipeline.semantic_anchor import SemanticAnchor, SemanticAnchorBuilder
from src.utils import is_chinese

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PropositionValidationResult:
    """NLI result for a single proposition."""

    proposition: Proposition
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    status: str  # "preserved" | "contradicted" | "missing"

    def __repr__(self) -> str:
        return (
            f"PropositionValidationResult("
            f"status={self.status!r}, "
            f"entailment={self.entailment_score:.3f}, "
            f"contradiction={self.contradiction_score:.3f}, "
            f"text={self.proposition.text[:40]!r}"
            f")"
        )


@dataclass
class ValidationResult:
    """Aggregated result of all three validation layers."""

    passed: bool
    failure_reason: str  # "" when passed=True
    layer1_similarity: float
    layer1_passed: bool
    layer2_results: list[PropositionValidationResult]
    contradicted_propositions: list[Proposition]
    missing_propositions: list[Proposition]
    layer3_bertscore: float | None  # BERTScore F1; None if unavailable
    rewrite_action: str  # "accept" | "retry_semantic"

    def __repr__(self) -> str:
        return (
            f"ValidationResult("
            f"passed={self.passed}, "
            f"layer1_similarity={self.layer1_similarity:.4f}, "
            f"contradicted={len(self.contradicted_propositions)}, "
            f"missing={len(self.missing_propositions)}, "
            f"action={self.rewrite_action!r}"
            f")"
        )


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------


class SemanticValidator:
    """
    Three-layer semantic validator for rewritten text.

    Parameters
    ----------
    anchor_builder : SemanticAnchorBuilder
        Pre-instantiated builder that provides layer-1 similarity checking.
    llm_client : LLMClient, optional
        Reserved for future LLM-based yes/no fallback; not used currently.
    nli_model_name : str
        HuggingFace model identifier for the NLI cross-encoder.
    layer1_threshold : float
        Minimum cosine similarity for layer-1 to pass (default 0.80).
    entailment_threshold : float
        Minimum entailment probability to classify a proposition as
        "preserved" (default 0.75).
    contradiction_threshold : float
        Minimum contradiction probability to classify a proposition as
        "contradicted" (default 0.70).
    max_missing_allowed : int
        Maximum number of missing propositions that still allow the
        rewrite to be accepted.  When missing > max_missing_allowed the
        validation fails.  Default 1 (i.e. ≥ 2 missing triggers retry).
    """

    def __init__(
        self,
        anchor_builder: SemanticAnchorBuilder,
        llm_client=None,
        nli_model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        layer1_threshold: float = 0.80,
        entailment_threshold: float = 0.75,
        contradiction_threshold: float = 0.70,
        max_missing_allowed: int = 1,
    ) -> None:
        self._anchor_builder = anchor_builder
        self._llm_client = llm_client  # reserved for future use
        self._nli_model_name = nli_model_name
        self._layer1_threshold = layer1_threshold
        self._entailment_threshold = entailment_threshold
        self._contradiction_threshold = contradiction_threshold
        self._max_missing_allowed = max_missing_allowed

        # Importance-weighted thresholds for Layer 2 NLI.
        # High-importance propositions require stricter entailment to be
        # considered "preserved"; low-importance ones are more forgiving.
        # Format: importance -> (entailment_threshold, contradiction_threshold)
        self._importance_thresholds: dict[str, tuple[float, float]] = {
            "high": (0.85, 0.65),
            "medium": (0.75, 0.70),
            "low": (0.65, 0.75),
        }

        # Lazy-loaded NLI artefacts
        self._nli_tokenizer = None
        self._nli_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        anchor: SemanticAnchor,
        rewritten_text: str,
        propositions: list[Proposition],
    ) -> ValidationResult:
        """
        Run all three validation layers and return a consolidated result.

        Parameters
        ----------
        anchor : SemanticAnchor
            Anchor built from the original (source) text.
        rewritten_text : str
            Candidate rewritten document to validate.
        propositions : list[Proposition]
            Propositions extracted from the original text (step 1 output).

        Returns
        -------
        ValidationResult
        """
        # ---- Layer 1 -------------------------------------------------------
        l1_similarity, l1_passed = self._layer1_check(anchor, rewritten_text)

        # ---- Layer 2 -------------------------------------------------------
        l2_results = self._layer2_check(rewritten_text, propositions)

        contradicted: list[Proposition] = [
            r.proposition for r in l2_results if r.status == "contradicted"
        ]
        missing: list[Proposition] = [r.proposition for r in l2_results if r.status == "missing"]

        # ---- Layer 3 -------------------------------------------------------
        l3_score = self._layer3_check(anchor.original_text, rewritten_text)

        # ---- Aggregate pass/fail -------------------------------------------
        failure_reasons: list[str] = []
        if not l1_passed:
            failure_reasons.append(
                f"Layer 1 similarity {l1_similarity:.4f} below threshold "
                f"{self._layer1_threshold:.2f}"
            )
        if contradicted:
            failure_reasons.append(f"Layer 2: {len(contradicted)} proposition(s) contradicted")
        if len(missing) > self._max_missing_allowed:
            failure_reasons.append(
                f"Layer 2: {len(missing)} proposition(s) missing "
                f"(max allowed: {self._max_missing_allowed})"
            )

        passed = len(failure_reasons) == 0
        failure_reason = "; ".join(failure_reasons)

        # ---- Determine rewrite action --------------------------------------
        rewrite_action = self._decide_action(
            l1_passed=l1_passed,
            contradicted=contradicted,
            missing=missing,
        )

        return ValidationResult(
            passed=passed,
            failure_reason=failure_reason,
            layer1_similarity=l1_similarity,
            layer1_passed=l1_passed,
            layer2_results=l2_results,
            contradicted_propositions=contradicted,
            missing_propositions=missing,
            layer3_bertscore=l3_score,
            rewrite_action=rewrite_action,
        )

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _layer1_check(
        self,
        anchor: SemanticAnchor,
        rewritten_text: str,
    ) -> tuple[float, bool]:
        """
        Layer 1: Coarse full-document similarity check.

        Returns
        -------
        (similarity, passed)
        """
        result = self._anchor_builder.check_similarity(anchor, rewritten_text)
        similarity: float = float(result["similarity"])
        passed: bool = bool(result["passed"])
        logger.debug("Layer 1 — similarity=%.4f, passed=%s", similarity, passed)
        return similarity, passed

    def _layer2_check(
        self,
        rewritten_text: str,
        propositions: list[Proposition],
    ) -> list[PropositionValidationResult]:
        """
        Layer 2: Proposition-level NLI verification.

        All propositions are verified in a single forward pass (batch) for
        efficiency. The rewritten_text is used as the NLI *premise* and each
        proposition text is used as a *hypothesis*.

        Returns
        -------
        list[PropositionValidationResult]
        """
        if not propositions:
            return []

        hypotheses = [prop.text for prop in propositions]
        all_scores = self._nli_check_batch(rewritten_text, hypotheses)

        results: list[PropositionValidationResult] = []
        for prop, scores in zip(propositions, all_scores, strict=True):
            entailment = scores.get("entailment", 0.0)
            contradiction = scores.get("contradiction", 0.0)
            neutral = scores.get("neutral", 0.0)

            # Use importance-weighted thresholds
            ent_t, con_t = self._importance_thresholds.get(
                prop.importance, (self._entailment_threshold, self._contradiction_threshold)
            )

            if entailment >= ent_t:
                status = "preserved"
            elif contradiction >= con_t:
                status = "contradicted"
            else:
                status = "missing"

            logger.debug(
                "Layer 2 prop=%r importance=%s entailment=%.3f(>=%.2f) "
                "contradiction=%.3f(>=%.2f) status=%s",
                prop.text[:60],
                prop.importance,
                entailment,
                ent_t,
                contradiction,
                con_t,
                status,
            )

            results.append(
                PropositionValidationResult(
                    proposition=prop,
                    entailment_score=entailment,
                    contradiction_score=contradiction,
                    neutral_score=neutral,
                    status=status,
                )
            )

        return results

    def _get_layer2_thresholds(self, importance: str) -> tuple[float, float]:
        """
        Return the (entailment_threshold, contradiction_threshold) to use for
        a proposition of the given *importance* level.
        """
        return self._importance_thresholds.get(
            importance,
            (self._entailment_threshold, self._contradiction_threshold),
        )

    def _nli_check_batch(self, premise: str, hypotheses: list[str]) -> list[dict]:
        """
        Run NLI inference for one premise against multiple hypotheses in a single
        forward pass (batch processing).

        The NLI model is loaded lazily on first call and cached on the instance
        for subsequent calls.

        Parameters
        ----------
        premise : str
            The rewritten text (full document), used as the NLI premise for
            all hypotheses.
        hypotheses : list[str]
            List of proposition texts to verify against the premise.

        Returns
        -------
        list[dict]
            One dict per hypothesis with keys "entailment", "contradiction",
            "neutral" (float each, summing to ≈ 1.0).
        """
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._nli_tokenizer is None or self._nli_model is None:
            logger.info(
                "Loading NLI model %s (first call — may download ~70 MB) …",
                self._nli_model_name,
            )
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self._nli_model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                self._nli_model_name
            )
            self._nli_model.eval()
            logger.info("NLI model loaded.")

        # Batch-tokenise all (premise, hypothesis) pairs
        inputs = self._nli_tokenizer(
            [premise] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = self._nli_model(**inputs).logits  # shape: (batch, 3)

        probs = torch.softmax(logits, dim=-1)  # shape: (batch, 3)
        id2label: dict = self._nli_model.config.id2label

        results: list[dict] = []
        for row in probs:
            scores = {id2label[i].lower(): row[i].item() for i in range(len(id2label))}
            results.append(
                {
                    "entailment": scores.get("entailment", 0.0),
                    "contradiction": scores.get("contradiction", 0.0),
                    "neutral": scores.get("neutral", 0.0),
                }
            )
        return results

    def _nli_check(self, premise: str, hypothesis: str) -> dict:
        """
        Run NLI inference for a single (premise, hypothesis) pair.

        Thin wrapper around _nli_check_batch for backward compatibility.
        """
        return self._nli_check_batch(premise, [hypothesis])[0]

    def _layer3_check(
        self,
        original_text: str,
        rewritten_text: str,
    ) -> float | None:
        """
        Layer 3: BERTScore word-coverage auxiliary check (F1 only).

        Returns None if bert_score is not installed or if any exception occurs.
        Failures here do NOT affect the overall validation outcome.

        Returns
        -------
        float | None  BERTScore F1 in [0, 1], or None on failure.
        """
        try:
            from bert_score import score as bert_score_fn  # type: ignore
        except ImportError:
            logger.debug("bert_score not installed; layer-3 check skipped.")
            return None

        try:
            # bert_score expects list inputs; we compare one pair
            P, R, F1 = bert_score_fn(
                cands=[rewritten_text],
                refs=[original_text],
                lang="zh" if is_chinese(rewritten_text) else "en",
                verbose=False,
            )
            f1_value = float(F1[0].item())
            logger.debug("Layer 3 BERTScore F1 = %.4f", f1_value)
            return f1_value
        except Exception as exc:  # noqa: BLE001
            logger.warning("Layer 3 BERTScore computation failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decide_action(
        self,
        l1_passed: bool,
        contradicted: list[Proposition],
        missing: list[Proposition],
    ) -> str:
        """
        Determine the rewrite action based on validation outcomes.

        Decision priority (highest to lowest):
        1. Any contradicted proposition    → "retry_semantic"
        2. Too many missing propositions   → "retry_semantic"
        3. Layer 1 failed                  → "retry_semantic"
        4. All checks passed               → "accept"
        """
        if contradicted:
            return "retry_semantic"
        if len(missing) > self._max_missing_allowed:
            return "retry_semantic"
        if not l1_passed:
            return "retry_semantic"
        return "accept"
