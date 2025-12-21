import os
from typing import Dict, Any
from app.config import settings
from app.vectorstore import get_vectorstore, build_chroma_from_documents
from app.rag.loaders import discover_pdfs, load_pdfs
from app.rag.splitter import split_docs
from langchain_chroma import Chroma


def index_docs(chunks, backend: str = "chroma") -> Dict[str, Any]:
    if not chunks:
        return {"chunks_indexed": 0, "status": "no_chunks"}
    debug = {}
    try:
        if backend == "chroma":
            store = build_chroma_from_documents(chunks)
            # diagnostics
            try:
                collection = getattr(store, "_collection", None)
                if collection:
                    debug["collection_name"] = getattr(collection, "name", "unknown")
                    debug["collection_count"] = collection.count() if hasattr(collection, "count") else "unknown"
            except Exception as e:
                debug["collection_error"] = str(e)
            # list persist dir contents
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

