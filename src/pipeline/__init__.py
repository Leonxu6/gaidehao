"""
ScholAI 6-step pipeline factory.
Provides ScholAIPipeline — the single entry point for end-to-end text humanisation.

Steps executed by the pipeline:
  1. PropositionExtractor  — extract core propositions from the original text
  2. SemanticAnchorBuilder — embed the original text as a semantic anchor
  3. RetryOrchestrator     — rewrite → validate → quality-check loop (up to max_retries)
     3a. ProbabilisticRewriter  (Step 3)
     3b. SemanticValidator      (Step 4)
     3c. QualityController      (Step 5)
     3d. CoherenceFixer         (Step 6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Thin configuration object for ScholAIPipeline.

    Parameters
    ----------
    deepseek_api_key : str
        API key for the DeepSeek service.
    max_retries : int
        Maximum rewrite attempts inside RetryOrchestrator.
    similarity_threshold : float
        Minimum cosine similarity for the layer-1 semantic check.
    min_quality_score : float
        Minimum G-Eval overall score accepted without a warning.
    """

    deepseek_api_key: str
    max_retries: int = 3
    similarity_threshold: float = 0.80
    min_quality_score: float = 3.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Unified result object returned by ScholAIPipeline.run().

    Fields
    ------
    original_text       : The text passed to pipeline.run().
    final_text          : The humanised output text.
    accepted            : True if all checks passed within max_retries.
    total_attempts      : Number of rewrite attempts made.
    warning             : Non-empty string when the pipeline hit max retries or
                          encountered a non-fatal issue.
    propositions_count  : Number of propositions extracted in Step 1.
    layer1_similarity   : Cosine similarity from the last attempt's layer-1
                          check, or None if unavailable.
    final_quality_score : G-Eval overall score of the final text, or None if
                          unavailable.
    """

    original_text: str
    final_text: str
    accepted: bool
    total_attempts: int
    warning: str

    # Optional debug / verbose fields
    propositions_count: int = 0
    layer1_similarity: Optional[float] = None
    final_quality_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ScholAIPipeline:
    """
    End-to-end 6-step text humanisation pipeline.

    Usage
    -----
    ::

        pipeline = ScholAIPipeline.from_config(
            PipelineConfig(deepseek_api_key="sk-...")
        )
        result = pipeline.run("AI生成的文本...")
        print(result.final_text)

    The pipeline is intentionally stateless across calls to run() — the same
    instance can be reused for multiple documents without side effects.
    """

    def __init__(
        self,
        proposition_extractor,
        anchor_builder,
        orchestrator,
    ) -> None:
        self._proposition_extractor = proposition_extractor
        self._anchor_builder = anchor_builder
        self._orchestrator = orchestrator

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "ScholAIPipeline":
        """
        Build all pipeline components from a PipelineConfig and return a
        fully wired ScholAIPipeline instance.

        Construction order
        ------------------
        1. LLMConfig(api_key)        → LLMClient
        2. EmbeddingConfig()         → EmbeddingClient
        3. PropositionExtractor(llm)
        4. SemanticAnchorBuilder(emb, threshold)
        5. ProbabilisticRewriter(llm)
        6. SemanticValidator(anchor_builder)
        7. QualityController(llm)
        8. CoherenceFixer(rewriter)
        9. RetryOrchestrator(rewriter, validator, quality_controller,
                             coherence_fixer, max_retries, min_quality_score)
        """
        # Lazy imports keep startup fast and allow the module to be imported
        # in environments where not all dependencies are installed.
        from src.config import LLMConfig, EmbeddingConfig
        from src.models.llm_client import LLMClient
        from src.models.embedding_client import EmbeddingClient
        from src.pipeline.proposition_extractor import PropositionExtractor
        from src.pipeline.semantic_anchor import SemanticAnchorBuilder
        from src.pipeline.rewriter import ProbabilisticRewriter
        from src.pipeline.semantic_validator import SemanticValidator
        from src.pipeline.quality_controller import QualityController
        from src.pipeline.coherence_fixer import CoherenceFixer, RetryOrchestrator

        # 1. LLM client
        llm_config = LLMConfig(api_key=config.deepseek_api_key)
        llm_config.max_retries = config.max_retries
        llm_client = LLMClient(config=llm_config)

        # 2. Embedding client (lazy-loads BGE-M3 on first use)
        embedding_config = EmbeddingConfig()
        embedding_client = EmbeddingClient(config=embedding_config)

        # 3. Step 1 — Proposition extractor
        proposition_extractor = PropositionExtractor(
            llm_client=llm_client,
            config=llm_config,
        )

        # 4. Step 2 — Semantic anchor builder
        anchor_builder = SemanticAnchorBuilder(
            embedding_client=embedding_client,
            similarity_threshold=config.similarity_threshold,
        )

        # 5. Step 3 — Probabilistic rewriter
        rewriter = ProbabilisticRewriter(llm_client=llm_client)

        # 6. Step 4 — Semantic validator (reuses anchor_builder for layer-1)
        validator = SemanticValidator(
            anchor_builder=anchor_builder,
            layer1_threshold=config.similarity_threshold,
        )

        # 7. Step 5 — Quality controller
        quality_controller = QualityController(llm_client=llm_client)

        # 8. Step 6a — Coherence fixer
        coherence_fixer = CoherenceFixer(rewriter=rewriter)

        # 9. Step 6b — Retry orchestrator (drives steps 3-6)
        orchestrator = RetryOrchestrator(
            rewriter=rewriter,
            validator=validator,
            quality_controller=quality_controller,
            coherence_fixer=coherence_fixer,
            max_retries=config.max_retries,
            min_quality_score=config.min_quality_score,
        )

        return cls(
            proposition_extractor=proposition_extractor,
            anchor_builder=anchor_builder,
            orchestrator=orchestrator,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, text: str) -> PipelineResult:
        """
        Execute the full 6-step pipeline on *text*.

        Steps
        -----
        1. Extract propositions (PropositionExtractor.extract).
        2. Build semantic anchor (SemanticAnchorBuilder.build).
        3. Run the retry-orchestrated rewrite loop
           (RetryOrchestrator.run — internally covers Steps 3-6).
        4. Collect metadata and return a PipelineResult.

        Parameters
        ----------
        text : str
            The AI-generated text to humanise.

        Returns
        -------
        PipelineResult
        """
        if not text or not text.strip():
            return PipelineResult(
                original_text=text,
                final_text=text,
                accepted=False,
                total_attempts=0,
                warning="输入文本为空",
                propositions_count=0,
                layer1_similarity=None,
                final_quality_score=None,
            )

        logger.info("ScholAIPipeline.run() started (input length=%d chars).", len(text))

        # ------------------------------------------------------------------
        # Step 1: Proposition extraction
        # ------------------------------------------------------------------
        logger.debug("Step 1: extracting propositions …")
        extraction = self._proposition_extractor.extract(text)
        propositions = extraction.propositions
        logger.debug("Step 1: extracted %d proposition(s).", len(propositions))

        # ------------------------------------------------------------------
        # Step 2: Semantic anchor embedding
        # ------------------------------------------------------------------
        logger.debug("Step 2: building semantic anchor …")
        anchor = self._anchor_builder.build(text)
        logger.debug("Step 2: anchor built with %d chunk(s).", len(anchor.chunks))

        # ------------------------------------------------------------------
        # Step 3-6: Retry-orchestrated rewrite loop
        # ------------------------------------------------------------------
        logger.debug("Steps 3-6: starting retry orchestrator …")
        orch_result = self._orchestrator.run(
            text=text,
            anchor=anchor,
            propositions=propositions,
        )
        logger.info(
            "ScholAIPipeline.run() finished: accepted=%s, attempts=%d.",
            orch_result.accepted,
            orch_result.total_attempts,
        )

        # ------------------------------------------------------------------
        # Step 4: Collect metadata from the last attempt for the result
        # ------------------------------------------------------------------
        layer1_similarity: Optional[float] = None
        final_quality_score: Optional[float] = None

        if orch_result.attempts:
            last_attempt = orch_result.attempts[-1]

            # Layer-1 similarity from the last semantic validation
            if last_attempt.validation_result is not None:
                layer1_similarity = last_attempt.validation_result.layer1_similarity

            # Quality score from the last quality report
            if last_attempt.quality_report is not None:
                final_quality_score = last_attempt.quality_report.quality.overall_score

        return PipelineResult(
            original_text=text,
            final_text=orch_result.final_text,
            accepted=orch_result.accepted,
            total_attempts=orch_result.total_attempts,
            warning=orch_result.warning,
            propositions_count=len(propositions),
            layer1_similarity=layer1_similarity,
            final_quality_score=final_quality_score,
        )
