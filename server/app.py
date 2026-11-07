"""Optional thin HTTP API around the docqa pipeline, plus the static demo page.

Run from the repository root:

    uvicorn server.app:app --reload --port 8000

Then open http://127.0.0.1:8000/ for the demo page, or POST to /ask:

    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "How do I make a query parameter optional?"}'

The pipeline is loaded once at startup from DOCQA_CONFIG (default configs/default.json).
This server is not needed for evaluation; `python -m docqa ask` covers the CLI path.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from docqa.config import Config
from docqa.errors import DocQAError, EmptyQuestionError
from docqa.pipeline import DocQAPipeline

CLIENT_DIR = Path(__file__).resolve().parents[1] / "client"


class AskRequest(BaseModel):
    question: str = Field(..., description="A natural-language question about FastAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Config.load(os.environ.get("DOCQA_CONFIG", "configs/default.json"))
    app.state.pipeline = DocQAPipeline.from_config(cfg)
    app.state.settings = cfg.to_dict()
    yield
    app.state.pipeline = None


app = FastAPI(title="FastAPI Documentation Q&A", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    p = app.state.pipeline
    return {
        "status": "ok",
        "chunks": len(p.chunks),
        "retriever": p.retriever.name,
        "generator": p.generator.name,
        "model": p.generator.model_name,
    }


@app.post("/ask")
def ask(body: AskRequest) -> dict:
    try:
        record = app.state.pipeline.answer(body.question)
    except EmptyQuestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocQAError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return record.to_dict()


if CLIENT_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")
