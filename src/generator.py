from __future__ import annotations

import logging
from typing import Optional

from .config import FALLBACK_RESPONSE, GENERATOR_PROMPT, GENERATION_MODEL_NAME

_HAS_GEMINI = False
try:
    from langchain.llms import GoogleGemini
    _HAS_GEMINI = True
except ImportError:
    logging.warning("GoogleGemini LLM is unavailable; generator will use a fallback response.")


def generate_response(
    query: str,
    persona: str,
    intent: str,
    context: str,
    temperature: float = 0.2,
    max_output_tokens: int = 512,
) -> str:
    prompt = GENERATOR_PROMPT.format(
        query=query,
        persona=persona,
        intent=intent,
        context=context or "No relevant context was found."
    )

    if _HAS_GEMINI:
        try:
            llm = GoogleGemini(model_name=GENERATION_MODEL_NAME, temperature=temperature)
            return llm(prompt)
        except Exception as exc:
            logging.warning("Failed to generate response with Gemini: %s", exc)

    return _fallback_response(query=query, persona=persona, intent=intent, context=context)


def _fallback_response(query: str, persona: str, intent: str, context: str) -> str:
    if context:
        return (
            f"Hello {persona}, based on the support materials available, here is what I found:\n\n"
            f"{context[:1200].strip()}\n\n"
            "If you need more detail or the answer is not complete, I can escalate your request to a human support agent."
        )

    return FALLBACK_RESPONSE


if __name__ == "__main__":
    print(generate_response(
        "Can I get details on the company expense policy?",
        persona="employee",
        intent="policy_question",
        context="Employees must submit expense reports within 30 days of purchase."
    ))
