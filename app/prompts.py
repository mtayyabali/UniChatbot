import os

SYSTEM_INSTRUCTIONS = (
    "You are a university assistant. Use only the provided context to answer. "
    "If the answer is not found in the context, say you do not have enough information. "
    "Be concise and neutral."
)


def build_prompt(question: str, context_docs):
    # format context with numbered snippets and simple citation markers
    lines = ["System:", SYSTEM_INSTRUCTIONS, "", "Context:"]
    sources = []
    for i, d in enumerate(context_docs, start=1):
        md = getattr(d, "metadata", {}) or {}
        raw_file = md.get("file") or md.get("source")
        # derive a safe file basename
        file_name = os.path.basename(raw_file) if isinstance(raw_file, str) and raw_file else "unknown"
        page = md.get("page")
        page_str = str(page) if page is not None else "?"
        content = d.page_content or ""
        lines.append(f"[{i}] ({file_name} p.{page_str})\n{content}")
        sources.append({"file": raw_file or file_name, "page": page})
    lines.append("")
    lines.append("Question:")
    lines.append(question)
    prompt = "\n\n".join(lines)
    return prompt, sources
