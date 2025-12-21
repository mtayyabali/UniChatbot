from typing import Dict, Any
from app.vectorstore import as_retriever
from app.prompts import build_prompt, SYSTEM_INSTRUCTIONS

# For demo, we won't call an LLM for generation; we'll synthesize from retrieved context.
# You can later swap in a Cohere text generation model via LangChain if desired.


def answer_from_context(question: str, context_docs):
    if not context_docs:
        return (
            "I don't have enough information in the indexed documents to answer that. "
            "Please ingest more relevant PDFs (course syllabi, policies)."
        )
    # Simple heuristic: choose 1-2 best chunks and summarize minimally
    snippets = [d.page_content.strip() for d in context_docs[:2]]
    # produce a concise answer referencing sources implicitly
    return (
        f"Based on the available documents, here is what applies: \n\n"
        + "\n\n".join(snippets)
        + "\n\nIf you need more detail, refer to the cited sources."
    )


def chat_query(question: str, top_k: int) -> Dict[str, Any]:
    retriever = as_retriever(k=top_k)
    docs = retriever.get_relevant_documents(question)
    prompt, sources = build_prompt(question, docs)
    # generate answer grounded in retrieved docs
    answer = answer_from_context(question, docs)
    return {"answer": answer, "sources": sources}

