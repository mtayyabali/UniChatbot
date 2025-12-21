from typing import List


def answer_from_context(question: str, context_docs: List):
    if not context_docs:
        return (
            "I don't have enough information in the indexed documents to answer that. "
            "Please ingest more relevant PDFs (course syllabi, policies)."
        )
    snippets = [d.page_content.strip() for d in context_docs[:2] if d.page_content]
    return (
        f"Based on the available documents, here is what applies: \n\n"
        + "\n\n".join(snippets)
        + "\n\nIf you need more detail, refer to the cited sources."
    )

