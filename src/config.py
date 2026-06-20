from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
DB_DIR = BASE_DIR / "db"
CHROMA_PERSIST_DIR = DB_DIR / "chroma"
DB_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# Retriever settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", 4))

# Confidence thresholds
ESCALATION_CONFIDENCE_THRESHOLD = float(os.getenv("ESCALATION_CONFIDENCE_THRESHOLD", 0.65))
PERSONA_CONFIDENCE_THRESHOLD = float(os.getenv("PERSONA_CONFIDENCE_THRESHOLD", 0.45))

# Model and prompt settings
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "gemini-1.5-pro")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "textembedding-gecko")

CLASSIFIER_PROMPT = (
    "You are a support classifier. Analyze the user message and return a JSON object with the following keys: persona, intent, confidence. "
    "Persona should be one of: customer, employee, manager, hr, it_support. "
    "Intent should be one of: policy_question, technical_issue, billing_question, escalation_request, general_information. "
    "Confidence should be a number between 0.0 and 1.0."  
)

GENERATOR_PROMPT = (
    "You are a support assistant. Use the provided persona, intent, and context to answer the user query. "
    "If the information is not sufficient, offer a polite escalation recommendation.\n\n"
    "Context:\n{context}\n\n"
    "Persona: {persona}\n"
    "Intent: {intent}\n"
    "User question: {query}\n\n"
    "Answer in a concise, professional, persona-aware manner. "
)

FALLBACK_RESPONSE = (
    "I found the nearby support context, but I cannot answer confidently enough yet. "
    "I will escalate your request to a human support agent for a faster resolution."
)

# Database settings
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "supportdesk")
DB_CHARSET = "utf8mb4"

# Local storage fallback
SQLITE_DB_PATH = DB_DIR / "supportdesk.db"
