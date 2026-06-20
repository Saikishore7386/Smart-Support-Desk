from __future__ import annotations

import re
from typing import Dict

from .config import CLASSIFIER_PROMPT, PERSONA_CONFIDENCE_THRESHOLD

PERSONA_KEYWORDS = {
    "customer": ["customer", "client", "user", "account"],
    "employee": ["employee", "team", "staff", "colleague"],
    "manager": ["manager", "supervisor", "lead", "director"],
    "hr": ["hr", "human resources", "payroll", "benefits"],
    "it_support": ["it", "tech", "technical support", "help desk", "ticket"],
}

INTENT_KEYWORDS = {
    "policy_question": ["policy", "compliance", "procedure", "guideline"],
    "technical_issue": ["error", "issue", "bug", "cannot", "fail", "problem", "login"],
    "billing_question": ["billing", "invoice", "charge", "refund", "payment", "subscription"],
    "escalation_request": ["escalat", "urgent", "supervisor", "manager", "complain", "immediately"],
    "general_information": ["how", "what", "where", "when", "help", "information", "details"],
}


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def _match_best_category(text: str, catalog: dict[str, list[str]], default: str) -> tuple[str, float]:
    normalized = _normalize_text(text)
    best_category = default
    best_score = 0.0
    for category, keywords in catalog.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score > best_score:
            best_score = score
            best_category = category
    if best_score > 0:
        return best_category, min(1.0, best_score / 4.0 + PERSONA_CONFIDENCE_THRESHOLD)
    return default, PERSONA_CONFIDENCE_THRESHOLD


def classify_query(query: str) -> Dict[str, object]:
    """Return persona, intent, and confidence for a support query."""
    persona, persona_confidence = _match_best_category(query, PERSONA_KEYWORDS, "customer")
    intent, intent_confidence = _match_best_category(query, INTENT_KEYWORDS, "general_information")
    overall_confidence = min(1.0, (persona_confidence + intent_confidence) / 2.0)

    return {
        "persona": persona,
        "intent": intent,
        "confidence": overall_confidence,
        "metadata": {
            "prompt": CLASSIFIER_PROMPT,
            "persona_confidence": persona_confidence,
            "intent_confidence": intent_confidence,
        },
    }


if __name__ == "__main__":
    import json

    sample = "I need help with my vacation policy and whether my manager must approve it."
    print(json.dumps(classify_query(sample), indent=2))
