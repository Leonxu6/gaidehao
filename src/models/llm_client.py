"""
DeepSeek LLM API client.

DeepSeek exposes an OpenAI-compatible REST API, so we use the official
`openai` Python SDK with a custom base_url.

Features
--------
- generate()       : plain-text completion
- generate_json()  : structured JSON completion (response_format=json_object)
- Automatic retries with exponential back-off via tenacity
- Optional streaming (returns a generator of text chunks)
"""

from __future__ import annotations

import json
import logging
from typing import Generator, Optional

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from src.config import LLMConfig

logger = logging.getLogger(__name__)

# Exceptions that are safe to retry
_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


def _build_retry_decorator(cfg: LLMConfig):
    """Build a tenacity retry decorator from LLMConfig."""
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=cfg.retry_min_wait,
            max=cfg.retry_max_wait,
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class LLMClient:
    """
    Thin wrapper around the DeepSeek API (OpenAI-compatible).

    Parameters
    ----------
    config : LLMConfig
        Configuration object. If None, a default LLMConfig is created
        (requires DEEPSEEK_API_KEY in the environment).
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.cfg = config or LLMConfig()
        self._client = openai.OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,
            timeout=self.cfg.timeout,
        )
        # Build the retry decorator once so it captures self.cfg
        self._retry = _build_retry_decorator(self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a completion request and return the assistant's reply as a string.

        Parameters
        ----------
        prompt        : User message content.
        system_prompt : System message content.
        temperature   : Override the default temperature from config.
        max_tokens    : Override the default max_tokens from config.

        Returns
        -------
        str  The assistant's text response.
        """
        return self._retry(self._generate_once)(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant. Respond with valid JSON only.",
    ) -> dict:
        """
        Request a JSON-mode completion and parse the result into a dict.

        DeepSeek / OpenAI json_object mode guarantees the output is valid JSON.

        Parameters
        ----------
        prompt        : User message — should instruct the model what JSON to produce.
        system_prompt : System message — should mention JSON output.

        Returns
        -------
        dict  Parsed JSON response.
        """
        raw = self._retry(self._generate_json_once)(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return json.loads(raw)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming completion — yields text delta chunks as they arrive.

        Note: retries are NOT applied to streaming calls because the response
        body has already started arriving before a failure could be detected.
        Callers should wrap this in their own retry logic if needed.
        """
        messages = self._build_messages(system_prompt, prompt)
        stream = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            top_p=self.cfg.top_p,
            frequency_penalty=self.cfg.frequency_penalty,
            presence_penalty=self.cfg.presence_penalty,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _generate_once(
        self,
        prompt: str,
        system_prompt: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        """Single (non-retried) call to the completions endpoint."""
        messages = self._build_messages(system_prompt, prompt)
        response = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.cfg.temperature,
            top_p=self.cfg.top_p,
            frequency_penalty=self.cfg.frequency_penalty,
            presence_penalty=self.cfg.presence_penalty,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def _generate_json_once(self, prompt: str, system_prompt: str) -> str:
        """Single (non-retried) JSON-mode call."""
        messages = self._build_messages(system_prompt, prompt)
        response = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            frequency_penalty=self.cfg.frequency_penalty,
            presence_penalty=self.cfg.presence_penalty,
            max_tokens=self.cfg.max_tokens,
            response_format={"type": "json_object"},
            stream=False,
        )
        return response.choices[0].message.content or "{}"
