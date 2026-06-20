from __future__ import annotations

from .config import ESCALATION_CONFIDENCE_THRESHOLD

ESCALATION_KEYWORDS = ["escalat", "urgent", "supervisor", "manager", "complain", "immediately", "critical"]


def should_escalate(confidence: float, query: str, intent: str, answer: str) -> dict[str, object]:
    lower_query = query.lower()
    lower_answer = answer.lower()

    mention_escalation = any(keyword in lower_query for keyword in ESCALATION_KEYWORDS)
    low_confidence = confidence < ESCALATION_CONFIDENCE_THRESHOLD
    escalation_intent = intent == "escalation_request"
    unable_to_answer = "cannot" in lower_answer or "unable" in lower_answer or "escalate" in lower_answer

    escalate = low_confidence or mention_escalation or escalation_intent or unable_to_answer
    reasons: list[str] = []
    if escalation_intent:
        reasons.append("The user explicitly requested escalation.")
    if low_confidence:
        reasons.append("Classifier confidence is below threshold.")
    if mention_escalation:
        reasons.append("The query contains escalation-related language.")
    if unable_to_answer:
        reasons.append("The generated response indicated uncertainty.")

    if not reasons:
        reasons.append("The response met confidence and escalation criteria.")

    return {
        "escalate": escalate,
        "reason": " ".join(reasons),
        "confidence": confidence,
        "threshold": ESCALATION_CONFIDENCE_THRESHOLD,
    }


if __name__ == "__main__":
    print(should_escalate(0.55, "This is urgent, please escalate.", "general_information", "I am not sure."))
