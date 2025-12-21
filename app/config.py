import os
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Embeddings provider: 'openai' (default), 'ollama', or 'gemini'
    embeddings_provider: str = Field(default=os.getenv("EMBEDDINGS_PROVIDER", "openai"))

    # OpenAI embeddings + chat
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    openai_embed_model: str = Field(default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    openai_chat_model: str = Field(default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

    # Ollama embeddings (local)
    ollama_host: str = Field(default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_api_key: str = Field(default=os.getenv("OLLAMA_API_KEY", ""))
    ollama_embed_model: str = Field(default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    # Gemini embeddings + chat
    gemini_api_key: str = Field(default=os.getenv("GEMINI_API_KEY", ""))
    gemini_embed_model: str = Field(default=os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004"))
    gemini_chat_model: str = Field(default=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))

    # Optional Chroma API key (for remote Chroma deployments)
    chroma_api_key: str = Field(default=os.getenv("CHROMA_API_KEY", ""))
    chroma_persist_dir: str = Field(default=os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    pdfs_dir: str = Field(default=os.getenv("PDFS_DIR", "data/pdfs"))
    chunk_size: int = Field(default=int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = Field(default=int(os.getenv("CHUNK_OVERLAP", "200")))
    top_k: int = Field(default=int(os.getenv("TOP_K", "4")))

    # Weaviate (optional remote vector DB)
    weaviate_host: str = Field(default=os.getenv("WEAVIATE_HOST", ""))
    weaviate_api_key: str = Field(default=os.getenv("WEAVIATE_API_KEY", ""))

settings = Settings()
