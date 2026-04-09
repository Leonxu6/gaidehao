"""
Step 3: Probabilistic Layer Rewriter

Rewrites input text so that its statistical characteristics (perplexity,
burstiness, TTR) shift toward the human-writing distribution, while strictly
preserving all core propositions extracted in Step 1.

Design choices
--------------
- Uses DeepSeek API with temperature=0.9 / top_p=0.85 / frequency_penalty=0.3
  to approximate Mirostat v2 sampling behaviour.
- Per-paragraph temperature variance simulates the natural rhythm fluctuations
  that Mirostat v2 would otherwise produce token-by-token.
- Three attempt modes change prompt strategy rather than sampling parameters:
    attempt=1 → standard rewrite
    attempt=2 → detection-evasion failure → emphasise sentence diversity
    attempt=3 → semantic failure → remind model of the dropped propositions
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from src.config import LLMConfig
from src.models.llm_client import LLMClient
from src.pipeline.proposition_extractor import Proposition, _is_chinese

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RewriteConfig:
    """Sampling parameters for the probabilistic rewriting step."""

    temperature: float = 0.9
    top_p: float = 0.85
    frequency_penalty: float = 0.3
    presence_penalty: float = 0.1
    max_tokens: int = 4096

    # Per-paragraph temperature variance.
    # Each paragraph is generated with temperature drawn from
    # Uniform(temperature - variance, temperature + variance),
    # clamped to [0.7, 1.1].  This simulates the natural burstiness
    # fluctuation that Mirostat v2 would produce at the token level.
    paragraph_temperature_variance: float = 0.1


@dataclass
class RewriteResult:
    """Result returned by ProbabilisticRewriter.rewrite()."""

    rewritten_text: str
    original_text: str
    propositions_included: list[str]   # proposition texts passed to the prompt
    config_used: RewriteConfig
    attempt_number: int                # 1 / 2 / 3


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ProbabilisticRewriter:
    """
    Probabilistic-layer text rewriter (Pipeline Step 3).

    Parameters
    ----------
    llm_client : LLMClient
        Instantiated DeepSeek API wrapper.
    config : RewriteConfig, optional
        Sampling configuration.  Defaults to RewriteConfig().
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[RewriteConfig] = None,
    ) -> None:
        self._client = llm_client
        self._config = config or RewriteConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(
        self,
        text: str,
        propositions: Optional[list[Proposition]] = None,
        attempt: int = 1,
    ) -> RewriteResult:
        """
        Probabilistic-layer rewrite of *text*.

        Parameters
        ----------
        text         : Source text to rewrite.
        propositions : Propositions from Step 1 used to constrain semantics.
        attempt      : Retry counter (1–3).  Controls prompt strategy.

        Returns
        -------
        RewriteResult
        """
        propositions = propositions or []

        # Split into paragraphs; use paragraph-variance path when there are
        # multiple paragraphs so we get natural burstiness fluctuation.
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

        if len(paragraphs) > 1:
            rewritten = self._apply_paragraph_variance(paragraphs, propositions, attempt)
        else:
            system_prompt, user_prompt = self._build_prompt(text, propositions, attempt)
            rewritten = self._call_with_temperature(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self._config.temperature,
            )

        included = [p.text for p in propositions if p.importance == "high"]

        return RewriteResult(
            rewritten_text=rewritten,
            original_text=text,
            propositions_included=included,
            config_used=self._config,
            attempt_number=attempt,
        )

    def rewrite_paragraph(
        self,
        paragraph: str,
        context: str = "",
        attempt: int = 1,
    ) -> str:
        """
        Rewrite a single paragraph, optionally with context from the
        preceding paragraph.  Used by Step 6 (coherence fixer) for
        targeted paragraph-level regeneration.

        Parameters
        ----------
        paragraph : The paragraph to rewrite.
        context   : Preceding paragraph text (optional).
        attempt   : Retry counter (influences prompt strategy).

        Returns
        -------
        str  The rewritten paragraph.
        """
        system_prompt, user_prompt = self._build_paragraph_prompt(
            paragraph=paragraph,
            context=context,
            attempt=attempt,
        )
        t = self._config.temperature
        return self._call_with_temperature(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=t,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        text: str,
        propositions: list[Proposition],
        attempt: int,
    ) -> tuple[str, str]:
        """
        Build (system_prompt, user_prompt) for a full-text rewrite.

        attempt=1 → standard rewrite
        attempt=2 → detection-evasion focus (more sentence diversity)
        attempt=3 → semantic-preservation focus (re-emphasise dropped propositions)
        """
        is_zh = _is_chinese(text)
        high_props = [p for p in propositions if p.importance == "high"]
        all_props = propositions  # include medium/low in the constraint list too

        if is_zh:
            system_prompt = self._zh_system_prompt(attempt)
            user_prompt = self._zh_user_prompt(text, all_props, high_props, attempt)
        else:
            system_prompt = self._en_system_prompt(attempt)
            user_prompt = self._en_user_prompt(text, all_props, high_props, attempt)

        return system_prompt, user_prompt

    def _build_paragraph_prompt(
        self,
        paragraph: str,
        context: str,
        attempt: int,
    ) -> tuple[str, str]:
        """Build prompts for single-paragraph rewrite (used by coherence fixer)."""
        is_zh = _is_chinese(paragraph)

        if is_zh:
            system_prompt = self._zh_system_prompt(attempt)
            context_block = f"\n上文（上一段）：\n{context}\n" if context else ""
            user_prompt = (
                f"{context_block}"
                f"\n请将下面这段文字用你自己的表达方式重新写出，"
                f"保持同样的核心意思，但语言风格自然、有个人特色：\n\n{paragraph}"
            )
        else:
            system_prompt = self._en_system_prompt(attempt)
            context_block = f"\nPreceding paragraph for context:\n{context}\n" if context else ""
            user_prompt = (
                f"{context_block}"
                f"\nRewrite the paragraph below in your own voice, "
                f"preserving its core meaning while making the language feel natural and personal:\n\n{paragraph}"
            )

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Language-specific prompt helpers
    # ------------------------------------------------------------------

    def _zh_system_prompt(self, attempt: int) -> str:
        base = (
            "你是一位有鲜明个人风格的中文写作者。"
            "你的任务是用自己的方式重新表达给定的内容，"
            "保留所有核心观点和事实，同时让文章读起来更像真实的人在写作。\n\n"
            "写作风格要求：\n"
            "1. 句子长度要有明显起伏——长句和短句交替出现，绝不要全是中等长度的句子。\n"
            "2. 偶尔使用稍微不那么常见的词汇或表达，但不要用生僻字。\n"
            "3. 每个段落可以用不同的方式开头，避免千篇一律的段落结构。\n"
            "4. 全部用散文形式，不要使用编号列表或项目符号。\n"
            "5. 不要使用\u201c首先\u201d、\u201c其次\u201d、\u201c此外\u201d、\u201c综上所述\u201d等套话连接词。\n"
            "6. 保持原文的语言（中文）。\n"
        )

        if attempt == 2:
            base += (
                "\n【特别提示 - 本次重写】\n"
                "上一次重写后的文章统计特征仍然太像AI生成的内容。\n"
                "请特别注意：\n"
                "- 大幅增加句子长度的变化幅度（有些句子要非常短，有些要很长）\n"
                "- 使用更多不寻常的句式结构和表达方式\n"
                "- 打破任何规律性的段落模式\n"
                "- 偶尔加入轻微的话语停顿感或思考感\n"
            )
        elif attempt == 3:
            base += (
                "\n【特别提示 - 本次重写】\n"
                "上一次重写遗漏了一些关键命题，语义验证未通过。\n"
                "请务必确保下面\u201c必须保留的命题\u201d部分列出的所有内容都明确出现在重写文章中。\n"
                "语义完整性优先于风格多样性。\n"
            )

        return base

    def _zh_user_prompt(
        self,
        text: str,
        all_props: list[Proposition],
        high_props: list[Proposition],
        attempt: int,
    ) -> str:
        parts: list[str] = []

        if high_props:
            parts.append("【必须保留的核心命题】")
            for i, p in enumerate(high_props, 1):
                parts.append(f"{i}. {p.text}")
            parts.append("")

        if all_props and attempt == 3:
            # On semantic-failure retry, list all props (not just high)
            medium_props = [p for p in all_props if p.importance in ("medium",)]
            if medium_props:
                parts.append("【同样需要体现的重要内容】")
                for i, p in enumerate(medium_props, 1):
                    parts.append(f"{i}. {p.text}")
                parts.append("")

        parts.append("【待重写的文本】")
        parts.append(text)
        parts.append("")
        parts.append(
            "请用你自己的语言重新表达以上内容，输出完整的重写版本，不要输出任何解释或说明。"
        )

        return "\n".join(parts)

    def _en_system_prompt(self, attempt: int) -> str:
        base = (
            "You are a writer with a distinctive personal voice. "
            "Your task is to re-express the given content in your own words, "
            "preserving every core argument and fact while making the prose feel "
            "authentically human.\n\n"
            "Style requirements:\n"
            "1. Vary sentence length dramatically — mix short punchy sentences with long, "
            "flowing ones. Never let all sentences settle into the same rhythm.\n"
            "2. Occasionally use less-common vocabulary or phrasing, but avoid obscure words.\n"
            "3. Open paragraphs in different ways; avoid formulaic paragraph structures.\n"
            "4. Use prose only — no numbered lists or bullet points.\n"
            "5. Avoid filler connectors like 'Furthermore', 'Moreover', 'In conclusion', "
            "'It is worth noting that'.\n"
            "6. Keep the language as the original (English).\n"
        )

        if attempt == 2:
            base += (
                "\n[Special instruction for this attempt]\n"
                "The previous rewrite still showed AI-typical statistical patterns.\n"
                "This time:\n"
                "- Exaggerate sentence-length variation even more\n"
                "- Use more unconventional sentence structures\n"
                "- Break any regular paragraph-level patterns\n"
                "- Let the prose breathe — add natural pauses and asides\n"
            )
        elif attempt == 3:
            base += (
                "\n[Special instruction for this attempt]\n"
                "The previous rewrite failed semantic validation — key propositions were dropped.\n"
                "Ensure every proposition listed under 'Must preserve' appears clearly "
                "in your rewrite. Semantic completeness takes priority over stylistic variety.\n"
            )

        return base

    def _en_user_prompt(
        self,
        text: str,
        all_props: list[Proposition],
        high_props: list[Proposition],
        attempt: int,
    ) -> str:
        parts: list[str] = []

        if high_props:
            parts.append("[Must preserve — core propositions]")
            for i, p in enumerate(high_props, 1):
                parts.append(f"{i}. {p.text}")
            parts.append("")

        if all_props and attempt == 3:
            medium_props = [p for p in all_props if p.importance == "medium"]
            if medium_props:
                parts.append("[Also required — important supporting points]")
                for i, p in enumerate(medium_props, 1):
                    parts.append(f"{i}. {p.text}")
                parts.append("")

        parts.append("[Text to rewrite]")
        parts.append(text)
        parts.append("")
        parts.append(
            "Rewrite the text above in your own voice. "
            "Output only the rewritten version — no explanations or commentary."
        )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Paragraph-level temperature variance (simulates Mirostat burstiness)
    # ------------------------------------------------------------------

    def _apply_paragraph_variance(
        self,
        paragraphs: list[str],
        propositions: list[Proposition],
        attempt: int = 1,
    ) -> str:
        """
        Rewrite each paragraph individually with a slightly varied temperature,
        then join the results.

        The variance creates natural rhythm fluctuations across the document,
        approximating the paragraph-level effect of Mirostat v2 sampling.

        Architecture note: propositions are included in the prompt for EVERY
        paragraph so the model never loses track of required semantics, even
        when generating a middle paragraph in isolation.  High-importance props
        are always listed; medium/low props are included only on attempt >= 3.
        """
        base_t = self._config.temperature
        variance = self._config.paragraph_temperature_variance
        rewritten_paragraphs: list[str] = []

        for para in paragraphs:
            # Sample per-paragraph temperature
            t = base_t + random.uniform(-variance, variance)
            t = max(0.7, min(1.1, t))

            system_prompt, user_prompt = self._build_prompt(para, propositions, attempt)
            rewritten_para = self._call_with_temperature(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=t,
            )
            rewritten_paragraphs.append(rewritten_para.strip())

        return "\n\n".join(rewritten_paragraphs)

    # ------------------------------------------------------------------
    # Low-level API call
    # ------------------------------------------------------------------

    def _call_with_temperature(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> str:
        """
        Call the LLM with the given temperature.

        We reach into LLMClient._generate_once via the existing generate()
        method, passing temperature as an override.  The other sampling
        parameters (top_p, frequency_penalty, presence_penalty) come from
        the LLMClient's underlying LLMConfig — we set those to match
        RewriteConfig at construction time.
        """
        return self._client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=self._config.max_tokens,
        )
