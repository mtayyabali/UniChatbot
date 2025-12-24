# UniChatbot

University Course & Policy Chatbot built with FastAPI + LangChain. Retrieval-augmented generation (RAG) over local university PDFs using configurable embeddings and vector databases.

## Key Features
- RAG pipeline: load, chunk, embed, store, retrieve, answer
- Pluggable embeddings: OpenAI, Ollama (local), Google Gemini
- Pluggable vector database: Chroma (local, persistent) or Weaviate (remote)
- Clean FastAPI endpoints for ingestion and chat
- Local PDF storage under `data/pdfs`

## Technology Stack
- Runtime & Frameworks:
  - Python 3.12
  - FastAPI (HTTP + WebSocket endpoints)
  - Uvicorn (ASGI server)
- RAG & ML:
  - LangChain core + community + text-splitters
  - Embeddings providers: OpenAI (`langchain-openai`), Google Gemini (`langchain-google-genai`), Ollama (local)
  - Vector stores: ChromaDB (local via `langchain-chroma`), Weaviate (remote via `langchain_community.vectorstores.Weaviate`)
  - Tokenization: `tiktoken`
  - ONNX Runtime (`onnxruntime`) for compatibility with ChromaDB dependencies
- Data & Persistence:
  - Local PDFs under `data/pdfs`
  - Local Chroma persistent directory under `data/chroma`
  - Remote Weaviate Cloud (recommended for free cloud persistence)
- API & Networking:
  - `requests`, `httpx` (transitive usage)
  - WebSockets via FastAPI/Starlette
- Observability & Utils:
  - `rich` for formatting (transitive via Chroma)
  - `pydantic` and `pydantic-settings` for config/validation
- Testing:
  - `pytest` (tests under `tests/`)
- Deployment:
  - Render (Docker runtime) with `Dockerfile`
  - Optional native deploy configs: `Procfile`, `render.yaml`, `railway.json` for alternative hosts

## Deployment

The backend (BE) is deployed on Render using the Docker runtime:
- Public base URL: https://unichatbot.onrender.com/
- Health check: `GET /health` → `https://unichatbot.onrender.com/health`
- REST APIs: `POST /ingest-pdfs`, `POST /chat`
- WebSocket (streaming): `WS /ws/chat` → `wss://unichatbot.onrender.com/ws/chat`

Notes
- The service is containerized with a `Dockerfile` and served via `uvicorn` bound to the Render-provided `$PORT`.
- WebSockets are supported by Render; the `/ws/chat` endpoint streams answer tokens and concludes with citations.
- For persistent vectors in the cloud, prefer Weaviate (set `WEAVIATE_HOST`, `WEAVIATE_API_KEY`). Local Chroma persistence is suitable for local dev.

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

### POST `/upload-pdfs`
- Upload one or more PDF files to the server; they will be saved under `PDFS_DIR` (default `data/pdfs`).
- Multipart form field name: `files`
- Response JSON includes saved/skipped files and destination directory.
- Example (two PDFs):
```bash
curl -s -X POST https://unichatbot.onrender.com/upload-pdfs \
  -H "Content-Type: multipart/form-data" \
  -F "files=@data/pdfs/sample-syllabus.pdf" \
  -F "files=@data/pdfs/Student-Handbook-2022-07.pdf" | python3 -m json.tool
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

## WebSocket Chat (streaming)

Real-time streaming responses are available via the WebSocket endpoint:

- Path: `GET /ws/chat` (WebSocket)
- Message format (client -> server, JSON):
  - `question` (string) – your question
  - `backend` (string, optional) – `chroma` (default) or `weaviate`
- Server sends:
  - First: `{"type":"sources","sources":[{file,page},...],"backend":"...","top_k":N}`
  - Then: streamed text chunks of the answer via `send_text`
  - Finally: citations appended as plain text and `{"type":"done"}` JSON

Example (Python client):
```python
import asyncio
import json
import websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/chat") as ws:
        await ws.send(json.dumps({
            "question": "What is the exam retake policy?",
            "backend": "weaviate"  # or "chroma"
        }))
        try:
            while True:
                msg = await ws.recv()
                # messages can be JSON (sources/done/errors) or text chunks
                if msg.startswith("{"):
                    print("JSON:", msg)
                else:
                    print(msg, end="", flush=True)
        except websockets.ConnectionClosed:
            pass

asyncio.run(main())
```

Notes
- Provider selection is controlled by `EMBEDDINGS_PROVIDER` (`openai` default, `gemini`, `ollama`).
- For `openai` streaming, set `OPENAI_API_KEY`; for `gemini`, set `GEMINI_API_KEY`.
- The server strictly grounds answers on retrieved context. If no context is found, it returns a fallback message.
- Cloud platforms like Render and Railway support WebSockets; ensure your service exposes the correct port and uses `uvicorn` with `--host 0.0.0.0 --port $PORT`.

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
