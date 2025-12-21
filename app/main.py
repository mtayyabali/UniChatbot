from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.config import settings
from app.vectorstore import reset_vectorstore, get_vectorstore, as_retriever
from app.api.ingest import router as ingest_router
from app.api.chat import router as chat_router
from app.rag.index import ingest_all
from app.rag.prompts import build_prompt
from app.rag.answer import answer_from_context

app = FastAPI(title="UniChatbot", version="0.1.0")
app.include_router(ingest_router)
app.include_router(chat_router)


class IngestRequest(BaseModel):
    force_reset: bool = False
    backend: Optional[str] = "chroma"


class ChatRequest(BaseModel):
    question: str
    backend: Optional[str] = "chroma"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chroma_dir": settings.chroma_persist_dir,
        "pdfs_dir": settings.pdfs_dir,
        "env": {
            "embed_model": settings.openai_embed_model,
            "openai_key_set": bool(settings.openai_api_key),
            "chroma_key_set": bool(settings.chroma_api_key),
        },
    }


@app.post("/ingest-pdfs")
def ingest(req: IngestRequest):
    backend = req.backend or "chroma"
    if req.force_reset:
        reset_vectorstore(backend=backend)
        # re-init to create clean store
        get_vectorstore(backend=backend)
    summary = ingest_all(backend=backend)
    return summary


@app.post("/chat")
def chat(req: ChatRequest):
    backend = req.backend or "chroma"
    if not req.question or not req.question.strip():
        return {"error": "Question must not be empty."}
    retriever = as_retriever(k=settings.top_k, backend=backend)
    docs = retriever.get_relevant_documents(req.question.strip())
    # reuse existing chat flow
    prompt, sources = build_prompt(req.question.strip(), docs)
    answer = answer_from_context(req.question.strip(), docs)
    return {"answer": answer, "sources": sources, "backend": backend, "top_k": settings.top_k}
