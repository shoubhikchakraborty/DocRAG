from pathlib import Path
from typing import Iterable
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger.custom_logger import CustomLogger
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException

log = CustomLogger().get_logger(__name__)

def load_documents(paths: Iterable[Path]):
    docs= []
    try:
        for p in paths:
            ext= p.suffix.lower()
            if ext == '.pdf':
                loader= PyPDFLoader(str(p))
            elif ext == '.docx':
                loader= Docx2txtLoader(str(p))
            elif ext == '.txt':
                loader= TextLoader(str(p), encoding='utf-8')
            else:
                log.warning("Unsupported documents skipped", path= str(p))
                continue
            docs.extend(loader.load())

        log.info("Documents loaded", count= len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error= str(e))
        raise DocumentPortalException("Error loadeding document", e) from e
    

