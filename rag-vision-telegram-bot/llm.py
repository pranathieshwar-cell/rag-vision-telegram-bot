from __future__ import annotations

from openai import OpenAI

from config import Settings


class LLMError(RuntimeError):
    pass


def generate_with_openai(prompt: str, settings: Settings) -> str:
    if not settings.openai_api_key:
        raise LLMError("OPENAI_API_KEY is required")

    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.responses.create(
            model=settings.openai_model,
            input=prompt,
        )
    except Exception as exc:  # Avoid tight coupling to SDK exception types
        raise LLMError(f"OpenAI API error: {exc}") from exc

    text = getattr(response, "output_text", "") or ""
    if not text.strip():
        raise LLMError("OpenAI returned an empty response")
    return text.strip()
