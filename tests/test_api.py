import os
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "paths" in data and "embeddings" in data and "vector_dbs" in data


def test_ingest_chroma():
    # Ensure PDFs dir exists; test runs should not fail if empty
    os.makedirs("data/pdfs", exist_ok=True)
    r = client.post("/api/ingest", json={"force_reset": True, "backend": "chroma"})
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "chroma"
    assert data["status"] in ("ok", "no_chunks")


def test_chat_chroma():
    # Ask a simple question; backend defaults to chroma if not specified
    r = client.post("/api/chat", json={"question": "What is the exam retake policy?", "backend": "chroma"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["backend"] == "chroma"


def test_ingest_weaviate_if_configured():
    host = os.getenv("WEAVIATE_HOST", "")
    key = os.getenv("WEAVIATE_API_KEY", "")
    if not host or not key:
        return
    r = client.post("/api/ingest", json={"force_reset": True, "backend": "weaviate"})
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "weaviate"
    assert data["status"] in ("ok", "no_chunks")

