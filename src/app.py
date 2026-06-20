from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .classifier import classify_query
from .db_manager import initialize_database, log_escalation, log_interaction
from .escalator import should_escalate
from .generator import generate_response
from .rag_pipeline import RagPipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_query(query: str, user_id: int | None = None) -> dict[str, Any]:
    """Run the full support pipeline for a single query."""
    initialize_database()

    classifier_result = classify_query(query)
    persona = classifier_result["persona"]
    intent = classifier_result["intent"]
    confidence = float(classifier_result["confidence"])

    rag = RagPipeline()
    context = rag.get_context(query)
    response = generate_response(query, persona, intent, context)
    escalation_decision = should_escalate(confidence, query, intent, response)

    interaction_id = log_interaction(
        query=query,
        response=response,
        persona=persona,
        intent=intent,
        confidence=confidence,
        context=context,
        user_id=user_id,
    )

    escalation_id = None
    if escalation_decision["escalate"]:
        escalation_id = log_escalation(
            interaction_id=interaction_id,
            reason=escalation_decision["reason"],
            metadata=json.dumps({
                "persona": persona,
                "intent": intent,
                "confidence": confidence,
            }),
        )

    return {
        "query": query,
        "persona": persona,
        "intent": intent,
        "confidence": confidence,
        "context": context,
        "response": response,
        "escalation": escalation_decision,
        "interaction_id": interaction_id,
        "escalation_id": escalation_id,
    }


if __name__ == "__main__":
    sample_query = "I need urgent help with a policy question and may need to escalate."
    result = process_query(sample_query)
    print(json.dumps(result, indent=2))
