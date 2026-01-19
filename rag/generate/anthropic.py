"""Anthropic Claude-based answer generator."""

import time

from tenacity import retry, stop_after_attempt, wait_exponential

from rag.generate.citations import CitationBuilder
from rag.generate.prompts import RAG_SYSTEM_PROMPT, format_context
from rag.models.generation import GeneratedAnswer
from rag.models.retrieval import RankedResult


class AnthropicGenerator:
    """Answer generator using Anthropic Claude models.

    Generates answers with citations from retrieved context.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        system_prompt: str | None = None,
    ):
        """Initialize Anthropic generator.

        Args:
            api_key: Anthropic API key (uses env var if None)
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
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "Anthropic generation requires 'anthropic' package"
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

        # Build user message
        user_message = f"""Context:
{formatted_context}

Question: {query}

Please provide a well-structured answer with citations using [1], [2], etc."""

        # Call Anthropic
        response = await self.client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        generation_time = (time.perf_counter() - start_time) * 1000

        # Extract answer
        answer_text = response.content[0].text

        # Extract citations
        _, citations = self._citation_builder.extract_citations(
            answer_text, context
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            model=self._model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
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
            Answer text chunks
        """
        formatted_context = format_context(context, max_chars=8000)

        user_message = f"""Context:
{formatted_context}

Question: {query}

Answer:"""

        async with self.client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
