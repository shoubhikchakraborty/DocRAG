import re
import uuid
from typing import Iterable
import shutil
from pathlib import Path
from multi_doc_chat.logger.custom_logger import CustomLogger
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException

log = CustomLogger().get_logger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

def save_uploaded_files(uploaded_files: Iterable, target_dir: Path):
    try:
        
        target_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for uf in uploaded_files:
            # get filename from common attributes
            name = getattr(uf, "filename", getattr(uf, "name", "file"))
            ext = Path(name).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue

            # Clean file name (only alphanum, dash, underscore)
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(name).stem).lower()
            fname = f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"
            out = target_dir / fname
            

            # Write to disk robustly
            with open(out, "wb") as f_out:
                # Case 1: object has .file which is a file-like (common with FastAPI UploadFile)
                if hasattr(uf, "file") and hasattr(uf.file, "read"):
                    # use streaming copy to avoid reading entire file into memory
                    uf.file.seek(0)
                    shutil.copyfileobj(uf.file, f_out)

                # Case 2: object exposes .read() directly
                elif hasattr(uf, "read"):
                    data = uf.read()
                    # memoryview -> bytes
                    if isinstance(data, memoryview):
                        data = data.tobytes()
                    # str -> bytes
                    if isinstance(data, str):
                        data = data.encode("utf-8")
                    # ensure bytes-like
                    if not isinstance(data, (bytes, bytearray)):
                        raise ValueError("Unsupported data type returned from read()")
                    f_out.write(data)

                # Case 3: object exposes getbuffer()
                else:
                    buf = getattr(uf, "getbuffer", None)
                    if callable(buf):
                        data = buf()
                        if isinstance(data, memoryview):
                            data = data.tobytes()
                        if isinstance(data, str):
                            data = data.encode("utf-8")
                        if not isinstance(data, (bytes, bytearray)):
                            raise ValueError("Unsupported buffer type")
                        f_out.write(data)
                    else:
                        raise ValueError("Unsupported uploaded file object")

            saved.append(out)
            log.info("File saved for ingestion", uploaded=name, saved_as=str(out))

        return saved

    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e
