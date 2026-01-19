"""Prompt templates for answer generation."""

# System prompt for RAG with citations
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context.
2. Include citations in your answer using [1], [2], etc. format to reference the sources.
3. If the context doesn't contain enough information to answer, say so clearly.
4. Be concise but thorough in your answers.
5. Do not make up information that is not in the context.

Each context chunk is numbered [1], [2], etc. Use these numbers to cite your sources."""

# User prompt template
RAG_USER_PROMPT = """Context:
{context}

Question: {query}

Please provide a well-structured answer with citations."""

# Simple prompt without elaborate instructions
RAG_SIMPLE_PROMPT = """Based on the following context, answer the question. Cite sources using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""

# Prompt for structured output
RAG_STRUCTURED_PROMPT = """You are a helpful assistant. Answer the question using the provided context.

Context:
{context}

Question: {query}

Respond with a JSON object containing:
- "answer": Your answer text with citations like [1], [2]
- "confidence": A number from 0 to 1 indicating confidence
- "sources_used": List of citation numbers used

JSON Response:"""


def format_context(
    context: list,
    max_chars: int = 8000,
    include_metadata: bool = True,
) -> str:
    """Format context for inclusion in prompts.

    Args:
        context: List of RankedResult or similar objects
        max_chars: Maximum characters for context
        include_metadata: Whether to include metadata

    Returns:
        Formatted context string
    """
    formatted_parts = []
    total_chars = 0

    for i, result in enumerate(context, start=1):
        # Build header
        header = f"[{i}]"

        if include_metadata:
            title = result.metadata.get("document_title") or result.metadata.get("title")
            if title:
                header += f" ({title})"

            section = result.metadata.get("section_title")
            if section:
                header += f" - {section}"

        # Get content
        content = result.content.strip()

        # Format entry
        entry = f"{header}\n{content}\n"

        # Check if we'd exceed limit
        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                # Truncate this entry
                content = content[: remaining - len(header) - 20] + "..."
                entry = f"{header}\n{content}\n"
            else:
                break

        formatted_parts.append(entry)
        total_chars += len(entry)

    return "\n".join(formatted_parts)


def build_rag_messages(
    query: str,
    context: list,
    system_prompt: str | None = None,
    max_context_chars: int = 8000,
) -> list[dict[str, str]]:
    """Build messages for chat completion with RAG.

    Args:
        query: User query
        context: Retrieved context
        system_prompt: Optional custom system prompt
        max_context_chars: Max characters for context

    Returns:
        List of message dicts for chat API
    """
    formatted_context = format_context(context, max_context_chars)

    user_content = RAG_USER_PROMPT.format(
        context=formatted_context,
        query=query,
    )

    return [
        {
            "role": "system",
            "content": system_prompt or RAG_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
