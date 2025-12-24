from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app.config import settings
from app.vectorstore import reset_vectorstore, get_vectorstore, as_retriever
from app.api.ingest import router as ingest_router
from app.api.chat import router as chat_router
from app.api.ws import router as ws_router
from app.api.upload import router as upload_router
from app.rag.index import ingest_all
from app.rag.prompts import build_prompt
from app.rag.answer import answer_from_context
import os

app = FastAPI(title="UniChatbot", version="0.1.0")
app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(ws_router)
app.include_router(upload_router)


class IngestRequest(BaseModel):
    force_reset: bool = False
    backend: Optional[str] = "chroma"


class ChatRequest(BaseModel):
    question: str
    backend: Optional[str] = "chroma"


def _dir_writable(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test = os.path.join(path, ".writable")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True
    except Exception:
        return False


@app.get("/health")
def health():
    chroma_dir = settings.chroma_persist_dir
    pdfs_dir = settings.pdfs_dir
    embeddings_provider = (settings.embeddings_provider or "openai").lower()
    return {
        "status": "ok",
        "paths": {
            "chroma_dir": chroma_dir,
            "pdfs_dir": pdfs_dir,
        },
        "embeddings": {
            "provider": embeddings_provider,
            "openai": {
                "model": settings.openai_embed_model,
                "chat_model": settings.openai_chat_model,
                "key_set": bool(settings.openai_api_key),
            },
            "ollama": {
                "host": settings.ollama_host,
                "model": settings.ollama_embed_model,
            },
            "gemini": {
                "model": settings.gemini_embed_model,
                "chat_model": settings.gemini_chat_model,
                "key_set": bool(settings.gemini_api_key),
            },
        },
        "vector_dbs": {
            "chroma": {
                "dir_exists": os.path.isdir(chroma_dir),
                "dir_writable": _dir_writable(chroma_dir),
            },
            "weaviate": {
                "host_set": bool(settings.weaviate_host),
                "api_key_set": bool(settings.weaviate_api_key),
            },
        },
    }


@app.post("/ingest-pdfs")
def ingest(req: IngestRequest):
    backend = req.backend or "chroma"
    if backend not in ("chroma", "weaviate"):
        backend = "chroma"
    if req.force_reset:
        reset_vectorstore(backend=backend)  # type: ignore[arg-type]
        # re-init to create clean store
        get_vectorstore(backend=backend)  # type: ignore[arg-type]
    summary = ingest_all(backend=backend)  # type: ignore[arg-type]
    return summary


@app.post("/chat")
def chat(req: ChatRequest):
    backend = req.backend or "chroma"
    if backend not in ("chroma", "weaviate"):
        backend = "chroma"
    if not req.question or not req.question.strip():
        return {"error": "Question must not be empty."}
    retriever = as_retriever(k=settings.top_k, backend=backend)  # type: ignore[arg-type]
    docs = retriever.get_relevant_documents(req.question.strip())
    # reuse existing chat flow
    prompt, sources = build_prompt(req.question.strip(), docs)
    answer = answer_from_context(req.question.strip(), docs)
    return {"answer": answer, "sources": sources, "backend": backend, "top_k": settings.top_k}
