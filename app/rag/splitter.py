from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings


def split_docs(docs):
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if c.page_content and c.page_content.strip()]

