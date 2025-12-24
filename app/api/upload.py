from fastapi import APIRouter, UploadFile, File
from fastapi import HTTPException
from typing import List
import os
from app.config import settings

router = APIRouter(tags=["upload"])

ALLOWED_CONTENT_TYPES = {"application/pdf"}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    try:
        # simple write check
        test = os.path.join(path, ".writable")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload directory not writable: {path} ({e})")


def _sanitize_filename(name: str) -> str:
    # basic sanitation to avoid path traversal and odd chars
    name = os.path.basename(name)
    return name.replace(" ", "_")


@router.post("/upload-pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    dest_dir = os.path.abspath(settings.pdfs_dir)
    _ensure_dir(dest_dir)

    saved = []
    skipped = []
    for f in files:
        ct = str((getattr(f, "content_type", "") or "").lower())
        raw_name = str(getattr(f, "filename", "") or "")
        filename = _sanitize_filename(raw_name)
        if not filename:
            skipped.append({"filename": raw_name, "reason": "missing filename"})
            continue
        # Allow by content-type or extension
        if (ct not in ALLOWED_CONTENT_TYPES) and (not filename.lower().endswith(".pdf")):
            skipped.append({"filename": filename, "reason": f"unsupported content-type: {ct}"})
            continue
        dest_path = os.path.join(dest_dir, filename)
        try:
            # stream chunks to disk to avoid large memory usage
            with open(dest_path, "wb") as out:
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            await f.close()
            saved.append({"filename": filename, "path": dest_path})
        except Exception as e:
            skipped.append({"filename": filename, "reason": str(e)})

    return {
        "saved_count": len(saved),
        "skipped_count": len(skipped),
        "saved": saved,
        "skipped": skipped,
        "pdfs_dir": dest_dir,
    }
