import os
import json
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.utils.model_loader import ModelLoader


class FaissManager:
    """
    A production-friendly wrapper around LangChain's FAISS vector store.

    - Keeps an 'ingested_meta.json' to avoid duplicate ingestion (idempotency).
    - Provides add, search (via manager.vs), delete-by-fingerprint, and delete-by-metadata.
    - Ensures FAISS index and metadata JSON remain consistent.
    """

    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        # Load existing metadata if present and valid
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        # Embeddings model loader (decoupled)
        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embedding()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        """Return True if FAISS index files exist on disk."""
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """
        Compute a stable fingerprint for a document chunk:
        - If metadata contains 'source' (or 'file_path') use 'source::row_id'
        - Else fallback to SHA256(text)
        """
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        """Persist `_meta` to disk as JSON."""
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None) -> FAISS:
        """
        Load an existing FAISS index if present, otherwise create a new one using provided texts+metadatas.
        Raises DocumentPortalException if no index exists and no texts supplied.
        """
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs

    def add_documents(self, docs: List[Document]) -> int:
        """
        Add documents to the FAISS index in an idempotent manner.
        Returns the number of documents actually added.
        """
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents().")

        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                # already ingested
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()

        return len(new_docs)

    def _get_all_docs(self) -> List[Document]:
        """
        Retrieve all documents currently stored in the FAISS store.

        Implementation detail: use a similarity_search with k equal to number of items.
        """
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before accessing documents.")
        # best-effort attempt to fetch all documents:
        try:
            # FAISS stores mapping; fallback if attribute doesn't exist
            total = len(getattr(self.vs, "index_to_docstore_id", []))
            if total == 0:
                # as a fallback, try a reasonable upper bound
                total = 10000
        except Exception:
            total = 10000
        return self.vs.similarity_search("", k=total)

    def delete_documents(self, keys: List[str]) -> int:
        """
        Delete documents from FAISS index by their fingerprint keys.
        Ensures ingested_meta.json is updated and saved.

        Returns:
            int: number of fingerprints removed (from metadata).
        """
        # Defensive: remove any keys present in metadata even if FAISS isn't loaded
        removed_count_from_meta_only = 0
        for k in keys:
            if k in self._meta["rows"]:
                # we don't delete here yet; accumulate and delete after FAISS rebuild for atomicity
                removed_count_from_meta_only += 0  # placeholder

        if self.vs is None:
            # No FAISS loaded — simply remove keys from metadata and persist
            removed = 0
            for k in keys:
                if k in self._meta["rows"]:
                    del self._meta["rows"][k]
                    removed += 1
            if removed:
                self._save_meta()
            return removed

        # Fetch all documents in the store
        all_docs = self._get_all_docs()
        if not all_docs:
            # Nothing in FAISS — just update metadata
            removed = 0
            for k in keys:
                if k in self._meta["rows"]:
                    del self._meta["rows"][k]
                    removed += 1
            if removed:
                self._save_meta()
            return removed

        kept_docs: List[Document] = []
        deleted_keys = set()

        for d in all_docs:
            fp = self._fingerprint(d.page_content, d.metadata or {})
            if fp in keys:
                deleted_keys.add(fp)
                continue
            kept_docs.append(d)

        # If nothing remains after deletion -> clear index files and metadata entries
        if not kept_docs:
            # clear FAISS in-memory handle & index files on disk (if present)
            self.vs = None
            for k in deleted_keys:
                self._meta["rows"].pop(k, None)
            # Also remove any user-passed keys that were in metadata but not found in FAISS
            for k in keys:
                self._meta["rows"].pop(k, None)
            # persist metadata
            self._save_meta()

            # attempt to remove index files to keep folder consistent
            try:
                for f in ("index.faiss", "index.pkl"):
                    p = self.index_dir / f
                    if p.exists():
                        p.unlink()
            except Exception:
                # best-effort; don't block deletion on file delete failure
                pass

            return len(deleted_keys)

        # Rebuild FAISS index from kept documents
        texts = [d.page_content for d in kept_docs]
        metadatas = [d.metadata for d in kept_docs]
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas)
        self.vs.save_local(str(self.index_dir))

        # Update metadata JSON: remove deleted fingerprints
        removed_count = 0
        for k in deleted_keys:
            if k in self._meta["rows"]:
                del self._meta["rows"][k]
                removed_count += 1

        # Defensive: remove any user-provided keys that might only exist in the metadata
        for k in keys:
            if k in self._meta["rows"] and k not in deleted_keys:
                del self._meta["rows"][k]
                removed_count += 1

        if removed_count:
            self._save_meta()

        return removed_count

    def delete_by_metadata(self, filters: Dict[str, Any]) -> int:
        """
        Delete documents matching key-value pairs in metadata.

        Example:
            delete_by_metadata({"source": "fileA.pdf"})
            delete_by_metadata({"source": "fileA.pdf", "row_id": 3})

        Returns:
            int: number of documents (fingerprints) deleted.
        """
        if self.vs is None:
            # Nothing loaded; still remove matching entries from metadata if any
            matching = []
            for key in list(self._meta["rows"].keys()):
                # Attempt to reconstruct metadata-like keys (best-effort) is not feasible here.
                # If you rely solely on metadata deletes, ensure FAISS is loaded first.
                pass
            return 0

        all_docs = self._get_all_docs()
        if not all_docs:
            return 0

        keys_to_delete: List[str] = []
        for d in all_docs:
            md = d.metadata or {}
            match = all(md.get(k) == v for k, v in filters.items())
            if match:
                keys_to_delete.append(self._fingerprint(d.page_content, md))

        if not keys_to_delete:
            return 0

        return self.delete_documents(keys_to_delete)



