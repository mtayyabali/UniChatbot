from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.config import settings
from app.vectorstore import as_retriever
from app.rag.prompts import build_prompt
from app.rag.answer import answer_from_context

router = APIRouter(prefix="/api", tags=["chat"])

class ChatRequest(BaseModel):
    question: str
    backend: Optional[str] = "chroma"

@router.post("/chat")
def chat(req: ChatRequest):
    backend = req.backend or "chroma"
    if not req.question or not req.question.strip():
        return {"error": "Question must not be empty."}
    retriever = as_retriever(k=settings.top_k, backend=backend)
    docs = retriever.get_relevant_documents(req.question.strip())
    prompt, sources = build_prompt(req.question.strip(), docs)
    answer = answer_from_context(req.question.strip(), docs)
    return {"answer": answer, "sources": sources, "backend": backend, "top_k": settings.top_k}

