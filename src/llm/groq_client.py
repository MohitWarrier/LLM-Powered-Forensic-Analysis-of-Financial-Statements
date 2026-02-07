import logging

from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


def _build_client() -> OpenAI | None:
    """Build the OpenAI client, or return None if no API key is configured."""
    if not settings.LLM_API_KEY:
        logger.warning("No LLM API key configured. Set OPENAI_API_KEY in .env to enable LLM features.")
        return None
    logger.info("LLM client initialized (base_url=%s, model=%s)", settings.LLM_BASE_URL, settings.LLM_MODEL)
    return OpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )


_client = _build_client()


def is_llm_available() -> bool:
    """Check if LLM is configured and available."""
    return _client is not None


def get_client() -> OpenAI:
    """Return the raw OpenAI client for advanced usage (e.g., tool calling)."""
    if _client is None:
        raise RuntimeError("LLM not configured. Set OPENAI_API_KEY in .env")
    return _client


def chat_with_groq(prompt: str, model: str = None) -> str:
    if _client is None:
        raise RuntimeError(
            "LLM is not configured. Add your Groq API key to .env file:\n"
            "OPENAI_API_KEY=your_groq_api_key_here"
        )
    model = model or settings.LLM_MODEL
    logger.info("Sending LLM request (model=%s, prompt_length=%d)", model, len(prompt))
    response = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    result = response.choices[0].message.content.strip()
    logger.info("LLM response received (length=%d)", len(result))
    return result
