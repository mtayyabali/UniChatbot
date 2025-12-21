import glob
import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader


def discover_pdfs(dir_path: str) -> List[str]:
    os.makedirs(dir_path, exist_ok=True)
    files = glob.glob(os.path.join(dir_path, "**", "*.pdf"), recursive=True)
    return [f for f in files if os.path.getsize(f) > 0]


def load_pdfs(paths: List[str]) -> Tuple[list, list]:
    docs = []
    errors = []
    if not paths:
        return docs, errors
    for p in paths:
        try:
            loader = PyPDFLoader(p)
            loaded = loader.load()
            for d in loaded:
                d.metadata = {**d.metadata, "file": p}
            docs.extend(loaded)
        except Exception as e:
            errors.append({"file": p, "error": str(e)})
    return docs, errors

