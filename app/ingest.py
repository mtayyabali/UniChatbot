import glob
import os
import time
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings
from app.vectorstore import get_vectorstore, build_chroma_from_documents
from langchain_chroma import Chroma


def discover_pdfs(dir_path: str) -> List[str]:
    os.makedirs(dir_path, exist_ok=True)
    files = glob.glob(os.path.join(dir_path, "**", "*.pdf"), recursive=True)
    return [f for f in files if os.path.getsize(f) > 0]


def load_pdfs(paths: List[str]):
    docs = []
    errors = []
    if not paths:
        return docs, errors
    for p in paths:
        try:
            loader = PyPDFLoader(p)
            loaded = loader.load()
            # ensure metadata has file path
            for d in loaded:
                d.metadata = {
                    **d.metadata,
                    "file": p,
                }
            docs.extend(loaded)
        except Exception as e:
            errors.append({"file": p, "error": str(e)})
    return docs, errors


def split_docs(docs):
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    # filter out empty/whitespace-only chunks
    return [c for c in chunks if c.page_content and c.page_content.strip()]


def index_docs(chunks, backend: str = "chroma") -> Dict[str, Any]:
    if not chunks:
        return {"chunks_indexed": 0, "status": "no_chunks"}
    debug = {}
    try:
        if backend == "chroma":
            # Build a persistent Chroma store directly from documents
            store = build_chroma_from_documents(chunks)

            # Force persist via underlying client (newer LangChain removed store.persist())
            try:
                client = getattr(store, "_client", None)
                if client:
                    # Chroma 0.4+ uses client.persist()
                    if hasattr(client, "persist"):
                        client.persist()
                        debug["persist_called"] = True
                    else:
                        debug["persist_called"] = False
                        debug["persist_note"] = "Client has no persist method"
                    debug["client_type"] = type(client).__name__

                    # Get actual persist directory from client
                    settings_obj = getattr(client, "_settings", None)
                    if settings_obj:
                        persist_path = getattr(settings_obj, "persist_directory", None)
                        if persist_path:
                            debug["client_persist_dir"] = persist_path
            except Exception as e:
                debug["persist_client_error"] = str(e)

            # Get collection info
            try:
                collection = getattr(store, "_collection", None)
                if collection:
                    debug["collection_name"] = getattr(collection, "name", "unknown")
                    debug["collection_count"] = collection.count() if hasattr(collection, "count") else "unknown"
            except Exception as e:
                debug["collection_error"] = str(e)

            # Diagnostics: list files and directories under persist dir
            try:
                persist_dir = os.path.abspath(settings.chroma_persist_dir)
                listing = []
                dir_listing = []
                for root, dirs, files in os.walk(persist_dir):
                    for d in dirs:
                        dir_listing.append(os.path.join(root, d))
                    for f in files:
                        listing.append(os.path.join(root, f))
                debug["persist_dir"] = persist_dir
                debug["persist_files"] = listing
                debug["persist_dirs"] = dir_listing
            except Exception as e:
                debug = {"persist_dir": settings.chroma_persist_dir, "persist_error": str(e)}
            return {"chunks_indexed": len(chunks), "status": "ok", "backend": backend, "debug": debug}
        else:
            # weaviate or other backends
            vs = get_vectorstore(backend)
            vs.add_documents(chunks)
            return {"chunks_indexed": len(chunks), "status": "ok", "backend": backend}
    except Exception as e:
        return {"chunks_indexed": 0, "status": "error", "error": str(e), "backend": backend}


def ingest_all(backend: str = "chroma") -> Dict[str, Any]:
    pdfs = discover_pdfs(settings.pdfs_dir)
    docs, errors = load_pdfs(pdfs)
    chunks = split_docs(docs)
    summary = index_docs(chunks, backend=backend)
    summary.update({
        "files_indexed": len(pdfs),
        "documents_loaded": len(docs),
        "chunks_produced": len(chunks),
        "errors": errors,
    })
    return summary
