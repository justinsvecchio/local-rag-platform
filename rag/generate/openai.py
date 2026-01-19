"""OpenAI-based answer generator."""

import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from rag.generate.citations import CitationBuilder
from rag.generate.prompts import RAG_SYSTEM_PROMPT, format_context
from rag.models.generation import GeneratedAnswer
from rag.models.retrieval import RankedResult


class OpenAIGenerator:
    """Answer generator using OpenAI models.

    Generates answers with citations from retrieved context.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        system_prompt: str | None = None,
    ):
        """Initialize OpenAI generator.

        Args:
            api_key: OpenAI API key (uses env var if None)
            model: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_prompt: Custom system prompt
        """
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt or RAG_SYSTEM_PROMPT
        self._client = None
        self._citation_builder = CitationBuilder()

    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "OpenAI generation requires 'openai' package"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        query: str,
        context: list[RankedResult],
    ) -> GeneratedAnswer:
        """Generate answer with citations.

        Args:
            query: User query
            context: Retrieved and ranked context

        Returns:
            Generated answer with citations
        """
        start_time = time.perf_counter()

        # Format context
        formatted_context = format_context(context, max_chars=8000)

        # Build messages
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nPlease provide a well-structured answer with citations.",
            },
        ]

        # Call OpenAI
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        generation_time = (time.perf_counter() - start_time) * 1000

        # Extract answer and token usage
        answer_text = response.choices[0].message.content
        usage = response.usage

        # Extract citations
        _, citations = self._citation_builder.extract_citations(
            answer_text, context
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            generation_time_ms=generation_time,
        )

    async def generate_streaming(
        self,
        query: str,
        context: list[RankedResult],
    ):
        """Generate answer with streaming.

        Args:
            query: User query
            context: Retrieved context

        Yields:
            Answer text chunks as they're generated
        """
        formatted_context = format_context(context, max_chars=8000)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        stream = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
