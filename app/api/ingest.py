from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.vectorstore import reset_vectorstore, get_vectorstore
from app.rag.index import ingest_all

router = APIRouter(prefix="/api", tags=["ingest"])

class IngestRequest(BaseModel):
    force_reset: bool = False
    backend: Optional[str] = "chroma"

@router.post("/ingest")
def ingest(req: IngestRequest):
    backend = req.backend or "chroma"
    if req.force_reset:
        reset_vectorstore(backend=backend)
        get_vectorstore(backend=backend)
    return ingest_all(backend=backend)

