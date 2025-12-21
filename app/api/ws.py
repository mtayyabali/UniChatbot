from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.config import settings
from app.vectorstore import as_retriever
from app.rag.prompts import SYSTEM_INSTRUCTIONS
from typing import List
import asyncio

router = APIRouter(tags=["ws"])

# LLM clients
from openai import OpenAI
import google.generativeai as genai


def _format_context(docs) -> str:
    lines: List[str] = []
    for i, d in enumerate(docs, start=1):
        md = getattr(d, "metadata", {}) or {}
        file = md.get("file") or md.get("source") or "unknown"
        page = md.get("page")
        page_str = str(page) if page is not None else "?"
        content = d.page_content or ""
        lines.append(f"[{i}] ({file} p.{page_str})\n{content}")
    return "\n\n".join(lines)


def _build_system_prompt() -> str:
    # Strengthen grounding rules
    return (
        SYSTEM_INSTRUCTIONS
        + "\n\nRules: Answer ONLY using the provided context snippets. Do not invent details. "
        + "If the answer cannot be derived from the context, respond: 'I don't have enough information.' "
        + "Prefer quoting or paraphrasing the relevant lines. Be concise."
    )


def _build_user_prompt(question: str, docs) -> str:
    ctx = _format_context(docs)
    return (
        "Use only the following context to answer the user's question. If the answer is not found, say you do not have enough information.\n\n"
        f"Context:\n{ctx}\n\nQuestion:\n{question}"
    )


def _citations_text(docs) -> str:
    lines: List[str] = ["\n\nCitations:"]
    for i, d in enumerate(docs, start=1):
        md = getattr(d, "metadata", {}) or {}
        file = md.get("file") or md.get("source") or "unknown"
        page = md.get("page")
        page_str = str(page) if page is not None else "?"
        lines.append(f"[{i}] {file} p.{page_str}")
    return "\n".join(lines)


@router.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_json()
        question = (msg.get("question") or "").strip()
        backend = (msg.get("backend") or "chroma").lower()
        if not question:
            await ws.send_json({"error": "Question must not be empty."})
            await ws.close()
            return
        retriever = as_retriever(k=settings.top_k, backend=backend)
        docs = retriever.get_relevant_documents(question)
        # Strict grounding: if no docs, immediately refuse
        if not docs:
            await ws.send_json({"answer": "I don't have enough information to answer that.", "sources": []})
            await ws.close()
            return
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(question, docs)

        # Send sources first
        sources = []
        for i, d in enumerate(docs, start=1):
            md = getattr(d, "metadata", {}) or {}
            file = md.get("file") or md.get("source") or "unknown"
            page = md.get("page")
            sources.append({"file": file, "page": page})
        await ws.send_json({"type": "sources", "sources": sources, "backend": backend, "top_k": settings.top_k})

        provider = (settings.embeddings_provider or "openai").lower()
        if provider == "gemini":
            if not settings.gemini_api_key:
                await ws.send_json({"error": "GEMINI_API_KEY not configured."})
                await ws.close()
                return
            genai.configure(api_key=settings.gemini_api_key)
            gem_model = settings.gemini_chat_model or "gemini-2.5-flash"
            if gem_model.startswith("models/"):
                gem_model = gem_model.split("/", 1)[1]
            model = genai.GenerativeModel(model_name=gem_model, system_instruction=system_prompt)
            try:
                response = model.generate_content(user_prompt, stream=True)
                for chunk in response:
                    txt = getattr(chunk, "text", None)
                    if txt:
                        await ws.send_text(txt)
                    await asyncio.sleep(0)
            except Exception as e:
                await ws.send_json({"error": str(e)})
                await ws.close()
                return
        else:
            if not settings.openai_api_key:
                await ws.send_json({"error": "OPENAI_API_KEY not configured."})
                await ws.close()
                return
            client = OpenAI(api_key=settings.openai_api_key)
            stream = client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    await ws.send_text(delta)
                await asyncio.sleep(0)

        # Append citations at the end of the streamed output
        await ws.send_text(_citations_text(docs))
        await ws.send_json({"type": "done"})
        await ws.close()
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await ws.send_json({"error": str(e)})
        finally:
            await ws.close()
