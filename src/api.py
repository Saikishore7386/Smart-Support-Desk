from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.app import process_query

app = FastAPI(title="Smart Support Desk API", version="1.0")


class QueryRequest(BaseModel):
    query: str
    user_id: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query_endpoint(payload: QueryRequest) -> dict[str, object]:
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="The query field must not be empty.")

    return process_query(payload.query, user_id=payload.user_id)
