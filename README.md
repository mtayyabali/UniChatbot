# UniChatbot

University Course & Policy Chatbot built with FastAPI + LangChain. Retrieval-augmented generation (RAG) over local university PDFs using configurable embeddings and vector databases.

## Key Features
- RAG pipeline: load, chunk, embed, store, retrieve, answer
- Pluggable embeddings: OpenAI, Ollama (local), Google Gemini
- Pluggable vector database: Chroma (local, persistent) or Weaviate (remote)
- Clean FastAPI endpoints for ingestion and chat
- Local PDF storage under `data/pdfs`

## Technology Stack
- Python, FastAPI
- LangChain
- Embeddings: OpenAI / Ollama / Google Gemini
- Vector DB: ChromaDB (local) / Weaviate (remote)

## Project Structure
- `app/main.py` – FastAPI app and endpoints
- `app/ingest.py` – PDF discovery, loading, splitting, indexing
- `app/vectorstore.py` – Embeddings + vector store backends (Chroma/Weaviate)
- `app/prompts.py` – Prompt builder and citation formatting
- `data/pdfs` – Place your PDFs here
- `data/chroma` – Local Chroma persistence (when using `backend=chroma`)

## Setup
1) Create env and install deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Copy env template and fill values
```bash
cp .env.example .env
```

## Environment Configuration
You can control embeddings provider, vector DB backend, and paths via `.env`.

General
- `PDFS_DIR` – directory of PDFs to ingest (default `data/pdfs`)
- `CHUNK_SIZE` – chunk size (default `1000`)
- `CHUNK_OVERLAP` – overlap between chunks (default `200`)
- `TOP_K` – number of chunks to retrieve (default `4`)

Embeddings provider
- `EMBEDDINGS_PROVIDER` – `openai` (default) | `ollama` | `gemini`

OpenAI (if `EMBEDDINGS_PROVIDER=openai`)
- `OPENAI_API_KEY` – your OpenAI API key
- `OPENAI_EMBED_MODEL` – embedding model (default `text-embedding-3-small`; try `text-embedding-3-large` for higher quality)

Ollama (if `EMBEDDINGS_PROVIDER=ollama`)
- `OLLAMA_HOST` – local Ollama server (default `http://127.0.0.1:11434`)
- `OLLAMA_EMBED_MODEL` – embedding model (default `nomic-embed-text`)
- Note: Ollama runs locally and does not require an API key; ensure the model is pulled:
```bash
ollama pull nomic-embed-text
```

Google Gemini (if `EMBEDDINGS_PROVIDER=gemini`)
- `GEMINI_API_KEY` – your Gemini API key
- `GEMINI_EMBED_MODEL` – embedding model (default `text-embedding-004`; normalized to `models/text-embedding-004`)

Vector DB: Chroma (local)
- `CHROMA_PERSIST_DIR` – local path for Chroma persistence (default `data/chroma`)
- Data is stored on disk under this folder when using `backend=chroma`.

Vector DB: Weaviate (remote)
- `WEAVIATE_HOST` – e.g. `https://<your-endpoint>.weaviate.cloud` (must include `https://`)
- `WEAVIATE_API_KEY` – API key for Weaviate
- Index/class naming is provider-aware (e.g., `UniversityDocGemini`).

## Endpoints

### GET `/health`
- Returns basic status and environment info.
- Example:
```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

### POST `/ingest-pdfs`
- Index PDFs from `PDFS_DIR` into the chosen vector DB backend.
- Request body:
  - `force_reset` (bool) – if true, resets the backend store before ingest
  - `backend` (string) – `chroma` (default) or `weaviate`
- Response contains summary and, for Chroma, a debug section with persistence path and files.
- Examples:
```bash
# Ingest into Chroma (local)
curl -s -X POST http://127.0.0.1:8000/ingest-pdfs \
  -H "Content-Type: application/json" \
  -d '{"force_reset": true, "backend": "chroma"}' | python3 -m json.tool

# Ingest into Weaviate (remote)
curl -s -X POST http://127.0.0.1:8000/ingest-pdfs \
  -H "Content-Type: application/json" \
  -d '{"force_reset": true, "backend": "weaviate"}' | python3 -m json.tool
```

### POST `/chat`
- Ask a question; the app retrieves top `TOP_K` chunks from the selected backend and synthesizes an answer.
- Request body:
  - `question` (string) – your question
  - `backend` (string, optional) – `chroma` (default) or `weaviate`
- Response contains `answer`, `sources` (file/page citations), and `backend`.
- Examples:
```bash
# Using Chroma
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the exam retake policy?", "backend": "chroma"}' | python3 -m json.tool

# Using Weaviate
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many credits are required to graduate?", "backend": "weaviate"}' | python3 -m json.tool
```

## Example Questions
- "What is the grading policy for CS101?"
- "How many credits are required to graduate?"
- "What is the exam retake policy?"
- "What are the academic probation rules?"
- "What is the attendance policy?"
- "What is the deadline to drop a course?"

## Operational Notes
- Ensure server is running:
```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- Place PDFs under `data/pdfs` (or set `PDFS_DIR` in `.env`).
- When switching embeddings provider (`EMBEDDINGS_PROVIDER`), re-run ingest with `force_reset: true` so the provider-specific namespace (collection/class) is fresh.
- Chroma local persistence is under `CHROMA_PERSIST_DIR`. If you don’t see files immediately, ensure you’ve restarted the server after changes and check nested directories using:
```bash
find "$(grep -E '^CHROMA_PERSIST_DIR=' .env | cut -d= -f2)" -maxdepth 3 -print
```
- Weaviate host must include `https://` and the API key must be set. Retrieval uses `nearVector` (embeds locally) and returns `file/page/source` metadata when available.

## Troubleshooting
- Missing PDFs or zero chunks: verify files are readable and not scanned-only; consider OCR loaders.
- Ollama model missing: pull the model (`ollama pull nomic-embed-text`).
- Gemini model name format: use `text-embedding-004` (auto-normalized to `models/text-embedding-004`).
- Weaviate nearText error: we force `by_text=False` (use `nearVector`).

## License
MIT (or your preferred license).
