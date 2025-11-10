import os
from typing import List, Optional, Sequence, Union
from pathlib import Path

from langchain_community.vectorstores import FAISS
# Using langchain.schema.Document is fine for 0.3.27, but we don't strictly require type-checking here.
from langchain.schema import Document

from multi_doc_chat.logger.custom_logger import CustomLogger

log = CustomLogger().get_logger(__name__)


class FaissManager:
    """
    FaissManager for langchain 0.3.x + langchain-community 0.3.x.
    - Creates/loads a FAISS index under self.index_dir
    - Indexes chunk-level texts + metadata
    - Supports adding chunk Documents after initial create
    - Returns an up-to-date LangChain FAISS vectorstore
    """

    def __init__(self, index_dir_path: Union[str, os.PathLike], model_loader):
        self.index_dir = str(index_dir_path)
        os.makedirs(self.index_dir, exist_ok=True)
        self.model_loader = model_loader
        self._vs: Optional[FAISS] = None
        log.info("FaissManager initialized", index_dir=self.index_dir)

    def _faiss_paths_simple(self) -> tuple[str, str]:
        # The common files produced by FAISS.save_local in these versions:
        return (os.path.join(self.index_dir, "index.faiss"), os.path.join(self.index_dir, "index.pkl"))

    def _index_files_exist(self) -> bool:
        faiss_path, pkl_path = self._faiss_paths_simple()
        return os.path.exists(faiss_path) and os.path.exists(pkl_path)

    def load_or_create(self, *, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None) -> FAISS:
        """
        Load existing FAISS index if present, otherwise create from texts + metadatas.
        Returns the FAISS vectorstore object.
        """
        embeddings = self.model_loader.load_embedding()

        if self._vs is not None:
            return self._vs

        # load existing if present
        if self._index_files_exist():
            try:
                log.info("Loading existing FAISS index from disk", index_dir=self.index_dir)
                self._vs = FAISS.load_local(self.index_dir, embeddings)
                return self._vs
            except Exception as e:
                log.warning("Failed to load existing FAISS index; will recreate", error=str(e))

        # create new index from provided texts
        if not texts:
            raise ValueError("No texts provided to create FAISS index")

        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        if len(metas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")

        log.info("Creating new FAISS index", index_dir=self.index_dir, count=len(texts))
        # Use the simple signature compatible with your versions
        self._vs = FAISS.from_texts(list(texts), embeddings, metadatas=metas)
        # persist
        try:
            self._vs.save_local(self.index_dir)
        except Exception as e:
            log.warning("Failed to save FAISS index to disk", error=str(e))

        return self._vs

    def add_documents(self, docs: Sequence[Document]) -> int:
        """
        Add chunk-level Document objects to the FAISS index.
        Returns number of documents added.
        """
        if not docs:
            return 0

        embeddings = self.model_loader.load_embedding()
        docs_list = list(docs)

        # If no existing index, create base from first doc
        if self._vs is None:
            first = docs_list[0]
            first_text = getattr(first, "page_content", "")
            first_meta = getattr(first, "metadata", {}) or {}
            if not first_text or not first_text.strip():
                raise ValueError("First document has empty text; cannot create a base index")
            self._vs = FAISS.from_texts([first_text], embeddings, metadatas=[first_meta])
            docs_to_add = docs_list[1:]
        else:
            docs_to_add = docs_list

        # convert to texts/metas and filter empty
        texts = [getattr(d, "page_content", "") for d in docs_to_add]
        metas = [getattr(d, "metadata", {}) or {} for d in docs_to_add]
        nonempty = [(t, m) for t, m in zip(texts, metas) if t and t.strip()]

        if not nonempty:
            log.info("No non-empty chunk texts to add")
            # persist current vs just in case
            try:
                if self._vs:
                    self._vs.save_local(self.index_dir)
            except Exception:
                pass
            return 0

        add_texts, add_metas = zip(*nonempty)

        # Most versions provide add_texts
        try:
            self._vs.add_texts(list(add_texts), metadatas=list(add_metas))
            added_count = len(add_texts)
        except Exception as e:
            log.warning("FAISS.add_texts failed; rebuilding index from new texts only", error=str(e))
            # Fallback: create a fresh index from new texts (safer than failing completely)
            try:
                self._vs = FAISS.from_texts(list(add_texts), embeddings, metadatas=list(add_metas))
                added_count = len(add_texts)
            except Exception as e2:
                log.error("Fallback rebuild failed during add_documents", error=str(e2))
                raise

        # persist
        try:
            self._vs.save_local(self.index_dir)
        except Exception as e:
            log.warning("Failed to save FAISS index after adding docs", error=str(e))

        log.info("Added documents to FAISS", added=added_count, index_dir=self.index_dir)
        return added_count

    def get_vectorstore(self) -> Optional[FAISS]:
        """
        Return the up-to-date vectorstore; try to load from disk if cached instance not present.
        """
        if self._vs is not None:
            return self._vs

        embeddings = self.model_loader.load_embedding()
        if self._index_files_exist():
            try:
                self._vs = FAISS.load_local(self.index_dir, embeddings)
            except Exception as e:
                log.error("Failed to load FAISS vectorstore from disk", error=str(e))
                return None
        return self._vs
