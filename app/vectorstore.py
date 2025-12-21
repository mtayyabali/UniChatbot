from typing import Optional, Literal, List
import os
import shutil
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Weaviate
import weaviate
from app.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

_embeddings: Optional[object] = None
_vectorstore_chroma: Optional[Chroma] = None
_vectorstore_weaviate: Optional[Weaviate] = None
_chroma_client: Optional[chromadb.ClientAPI] = None


def _provider_suffix() -> str:
    prov = (settings.embeddings_provider or "openai").lower()
    if prov in ("openai", "ollama", "gemini"):
        return prov
    return "openai"


def _chroma_collection_name() -> str:
    return f"university_docs_{_provider_suffix()}"


def _weaviate_class_name() -> str:
    m = {"openai": "OpenAI", "ollama": "Ollama", "gemini": "Gemini"}
    return f"UniversityDoc{m.get(_provider_suffix(), 'OpenAI')}"


def get_embeddings():
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    provider = (settings.embeddings_provider or "openai").lower()
    if provider == "ollama":
        _embeddings = OllamaEmbeddings(
            model=settings.ollama_embed_model,
            base_url=settings.ollama_host,
        )
    elif provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Please configure your .env.")
        model = settings.gemini_embed_model or "text-embedding-004"
        if not model.startswith("models/"):
            model = f"models/{model}"
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=settings.gemini_api_key,
        )
    else:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Please configure your .env.")
        _embeddings = OpenAIEmbeddings(
            model=settings.openai_embed_model,
            api_key=settings.openai_api_key,
        )
    return _embeddings


def _ensure_persist_dir():
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    try:
        test_path = os.path.join(settings.chroma_persist_dir, ".writable")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
    except Exception as e:
        raise RuntimeError(f"Chroma persist dir not writable: {settings.chroma_persist_dir} ({e})")


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _ensure_persist_dir()
        # Use PersistentClient for disk-backed storage
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def get_chroma_vectorstore() -> Chroma:
    global _vectorstore_chroma
    if _vectorstore_chroma is not None:
        return _vectorstore_chroma

    client = _get_chroma_client()
    try:
        _vectorstore_chroma = Chroma(
            client=client,
            collection_name=_chroma_collection_name(),
            embedding_function=get_embeddings(),
        )
        return _vectorstore_chroma
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Chroma vector store: {e}")


def build_chroma_from_documents(docs: List[Document]) -> Chroma:
    client = _get_chroma_client()
    store = Chroma.from_documents(
        documents=docs,
        client=client,
        embedding=get_embeddings(),
        collection_name=_chroma_collection_name(),
    )
    return store


def reset_chroma():
    global _vectorstore_chroma, _chroma_client
    try:
        shutil.rmtree(settings.chroma_persist_dir, ignore_errors=True)
    except Exception:
        pass
    _vectorstore_chroma = None
    _chroma_client = None


def get_weaviate_client() -> weaviate.WeaviateClient:
    if not settings.weaviate_host or not settings.weaviate_api_key:
        raise RuntimeError("WEAVIATE_HOST/WEAVIATE_API_KEY not set in env.")
    auth = weaviate.AuthApiKey(api_key=settings.weaviate_api_key)
    return weaviate.Client(settings.weaviate_host, auth_client_secret=auth)


def get_weaviate_vectorstore() -> Weaviate:
    global _vectorstore_weaviate
    if _vectorstore_weaviate is not None:
        return _vectorstore_weaviate
    client = get_weaviate_client()
    _vectorstore_weaviate = Weaviate(
        client=client,
        index_name=_weaviate_class_name(),
        text_key="text",
        embedding=get_embeddings(),
        by_text=False,  # ensure nearVector is used (embed locally), avoiding nearText errors
        attributes=["file", "page", "source"],  # request metadata back with results
    )
    return _vectorstore_weaviate


def get_vectorstore(backend: Literal["chroma", "weaviate"] = "chroma"):
    if backend == "weaviate":
        return get_weaviate_vectorstore()
    return get_chroma_vectorstore()


def reset_vectorstore(backend: Literal["chroma", "weaviate"] = "chroma"):
    global _vectorstore_weaviate
    if backend == "weaviate":
        try:
            client = get_weaviate_client()
            class_name = _weaviate_class_name()
            try:
                if hasattr(client, "collections"):
                    client.collections.delete(class_name)
                else:
                    client.schema.delete_class(class_name)
            except Exception:
                try:
                    client.schema.delete_class(class_name)
                except Exception:
                    pass
            _vectorstore_weaviate = None
        except Exception:
            _vectorstore_weaviate = None
        return
    reset_chroma()


def as_retriever(k: Optional[int] = None, backend: Literal["chroma", "weaviate"] = "chroma"):
    k = k or settings.top_k
    return get_vectorstore(backend).as_retriever(search_kwargs={"k": k})
