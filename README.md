# Smart Support Desk

A layered support assistant built with a classifier, RAG pipeline, response generator, escalator, and logging layer.

## Project Layout

- `Data/` — knowledge base files: `faq.txt`, `company_policy.md`, `help_guide.pdf`
- `src/` — core application modules
  - `config.py` — constants, prompts, thresholds, environment config
  - `classifier.py` — persona and intent detection logic
  - `rag_pipeline.py` — document ingestion, chunking, retrieval
  - `generator.py` — response generation and fallback behavior
  - `escalator.py` — escalation decision logic
  - `db_manager.py` — persistence for interactions and escalations
  - `app.py` — orchestration for the end-to-end pipeline
- `app.py` — Streamlit UI
- `requirements.txt` — dependency list
- `src/api.py` — FastAPI endpoint for programmatic access

## Setup

1. Create a `.env` file in the project root with database and API config.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Run the FastAPI server:
   ```bash
   uvicorn src.api:app --reload
   ```

## API

- `GET /health` — returns `{ "status": "ok" }`
- `POST /query` — accepts JSON `{ "query": "your question", "user_id": 123 }`

## Notes

- The pipeline uses fallback modes when `langchain`, `chromadb`, or `PyPDF2` are unavailable.
- MySQL is preferred when configured; otherwise the project falls back to SQLite.
- The root `app.py` offers a simple interface for sending support queries and viewing classification, context, and escalation decisions.
