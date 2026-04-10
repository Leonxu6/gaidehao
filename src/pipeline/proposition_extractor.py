"""
Step 1: Proposition Extraction

Extracts core propositions from input text using an LLM.
Propositions are categorised as:
  - claim  : the author's viewpoint, stance, or conclusion
  - entity : numbers, proper nouns, names, organisations
  - fact   : objectively stated factual content

Supports Chinese and English input with language-appropriate prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import LLMConfig
from src.models.llm_client import LLMClient
from src.utils import is_chinese

logger = logging.getLogger(__name__)

# Valid enum values — used for validation after parsing
_VALID_TYPES = {"claim", "entity", "fact"}
_VALID_IMPORTANCES = {"high", "medium", "low"}


@dataclass
class Proposition:
    text: str  # verbatim proposition text
    prop_type: str  # "claim" | "entity" | "fact"
    importance: str  # "high" | "medium" | "low"

    def __repr__(self) -> str:
        return (
            f"Proposition(type={self.prop_type!r}, "
            f"importance={self.importance!r}, text={self.text[:40]!r})"
        )


@dataclass
class ExtractionResult:
    propositions: list[Proposition]
    original_text: str
    model_used: str


def _is_chinese(text: str) -> bool:
    """Proxy kept for backward compatibility — prefer src.utils.is_chinese."""
    return is_chinese(text)


class PropositionExtractor:
    """
    Extract propositions from text using an LLM in JSON mode.

    Parameters
    ----------
    llm_client : LLMClient
        Instantiated client wrapping the DeepSeek API.
    config : LLMConfig, optional
        If provided, its ``model`` attribute is recorded in the result.
        Defaults to a fresh LLMConfig() if not given.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: LLMConfig | None = None,
    ) -> None:
        self._client = llm_client
        self._config = config or LLMConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract propositions from *text* and return an ExtractionResult.

        Steps
        -----
        1. Build a language-appropriate prompt.
        2. Call generate_json() to get a structured response from the LLM.
        3. Parse, validate, deduplicate, and filter the raw propositions.
        4. Return an ExtractionResult.
        """
        prompt = self._build_prompt(text)

        system_prompt = (
            "你是一个专业的文本分析助手，擅长提取文章中的核心命题。请严格按照JSON格式输出。"
            if _is_chinese(text)
            else "You are a professional text analysis assistant specialised in "
            "extracting core propositions from text. Respond with valid JSON only."
        )

        raw: dict = {}
        try:
            raw = self._client.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed during proposition extraction: %s", exc)
            # Return an empty result rather than crashing the pipeline
            return ExtractionResult(
                propositions=[],
                original_text=text,
                model_used=self._config.model,
            )

        propositions = self._parse_and_validate(raw)

        return ExtractionResult(
            propositions=propositions,
            original_text=text,
            model_used=self._config.model,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, text: str) -> str:
        """
        Build the extraction prompt, choosing Chinese or English wording
        based on the dominant language of *text*.
        """
        if _is_chinese(text):
            return self._build_chinese_prompt(text)
        return self._build_english_prompt(text)

    def _build_chinese_prompt(self, text: str) -> str:
        return f"""请仔细阅读以下文本，提取其中的核心命题。

命题分三类：
- claim（论点）：作者表达的核心观点、立场或结论
- entity（实体）：数字、专有名词、人名、机构名、地名等
- fact（事实）：客观陈述的事实性内容

重要程度分三级：
- high：对文章主旨至关重要
- medium：支撑主要论点的内容
- low：补充说明性的细节

请提取所有有意义的命题（通常5-15条），并按以下JSON格式输出：
{{
  "propositions": [
    {{"text": "命题原文或简洁概括", "type": "claim/entity/fact", "importance": "high/medium/low"}},
    ...
  ]
}}

注意：
1. text 字段不能为空
2. type 只能是 claim、entity、fact 之一
3. importance 只能是 high、medium、low 之一
4. 不同命题不要重复

待分析文本：
{text}"""

    def _build_english_prompt(self, text: str) -> str:
        return f"""Read the following text carefully and extract its core propositions.

Propositions fall into three categories:
- claim  : the author's core viewpoint, stance, or conclusion
- entity : numbers, proper nouns, names, organisations, places
- fact   : objectively stated factual content

Importance levels:
- high   : central to the main thesis
- medium : supports a key argument
- low    : supplementary detail

Extract all meaningful propositions (typically 5–15) and return them in this exact JSON format:
{{
  "propositions": [
    {{"text": "proposition text", "type": "claim/entity/fact", "importance": "high/medium/low"}},
    ...
  ]
}}

Rules:
1. "text" must not be empty
2. "type" must be one of: claim, entity, fact
3. "importance" must be one of: high, medium, low
4. No duplicate propositions

Text to analyse:
{text}"""

    def _parse_and_validate(self, raw: dict) -> list[Proposition]:
        """
        Parse the LLM JSON response into a deduplicated, validated list
        of Proposition objects.
        """
        raw_list = raw.get("propositions", [])
        if not isinstance(raw_list, list):
            logger.warning("Unexpected LLM response structure: %s", raw)
            return []

        seen_texts: set[str] = set()
        result: list[Proposition] = []

        for item in raw_list:
            if not isinstance(item, dict):
                continue

            text = (item.get("text") or "").strip()
            if not text:
                continue  # filter out empty propositions

            # Deduplication (case-insensitive)
            lower = text.lower()
            if lower in seen_texts:
                continue
            seen_texts.add(lower)

            # Validate / coerce type
            prop_type = str(item.get("type", "")).lower().strip()
            if prop_type not in _VALID_TYPES:
                logger.debug("Unknown prop_type %r — defaulting to 'fact'", prop_type)
                prop_type = "fact"

            # Validate / coerce importance
            importance = str(item.get("importance", "")).lower().strip()
            if importance not in _VALID_IMPORTANCES:
                logger.debug("Unknown importance %r — defaulting to 'medium'", importance)
                importance = "medium"

            result.append(Proposition(text=text, prop_type=prop_type, importance=importance))

        return result
