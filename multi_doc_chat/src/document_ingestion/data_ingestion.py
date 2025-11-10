import uuid
from pathlib import Path
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger.custom_logger import CustomLogger
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.utils.faiss_manager import FaissManager

log = CustomLogger().get_logger(__name__)

def generate_session_id():
    unique_id= uuid.uuid4().hex[:8]
    timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"session_{timestamp}_{unique_id}"

class ChatIngestor:
    def __init__(self, temp_base= 'data', 
                 faiss_base= 'faiss_index',
                 use_session_dirs=  True,
                 session_id= None):
        try:
            self.model_loader= ModelLoader()
            self.use_session= use_session_dirs
            self.session_id= session_id or generate_session_id()

            self.temp_base= Path(temp_base)
            self.temp_base.mkdir(parents= True, exist_ok= True)
            self.faiss_base= Path(faiss_base)
            self.faiss_base.mkdir(parents= True, exist_ok= True)

            self.temp_dir= self._resolve_dir(self.temp_base)
            self.faiss_dir= self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor Initialized",
                     session_id= self.session_id,
                     temp_dir= str(self.temp_dir),
                     faiss_dir= str(self.faiss_dir),
                     sessionized= self.use_session)
        except Exception as e:
            log.error("Failed to initiale ChatIngestor", error= str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e)
        
    def _resolve_dir(self, base):
        if self.use_session:
            d= base / self.session_id  #base Path for each session
            d.mkdir(parents= True, exist_ok=  True)
            return d
        return base
    
    def _split(self, docs, chunk_size= 250, chunk_overlap= 50):
        splitter= RecursiveCharacterTextSplitter(chunk_size= chunk_size,
                                                 chunk_overlap= chunk_overlap)
        chunks= splitter.split_documents(docs)
        log.info("Document Split", chunks= len(chunks), chunk_size= chunk_size, chunk_overlap= chunk_overlap)
        return chunks
    
    def build_retriver(self,
                   uploaded_files,
                   chunk_size=250,
                   chunk_overlap=50,
                   k=5,
                   search_type='mmr',
                   fetch_k=20,
                   lambda_mult=0.5):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            # Split into chunks (index and retrieval granularity)
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            log.info("Split documents", docs=len(docs), chunks=len(chunks),
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Debug prints - first 3 chunk samples for quick manual inspection
            #print(f"[DEBUG] total docs: {len(docs)}; total chunks: {len(chunks)}")
            for i, c in enumerate(chunks[:3], start=1):
                sample = getattr(c, "page_content", "")[:500].replace("\n", " ")
                #print(f"[DEBUG] chunk #{i} (len={len(sample)}): {sample[:300]}...")

            # Build texts + metadata from chunks (important: chunk-level, not doc-level)
            chunk_texts = [getattr(c, "page_content", "") for c in chunks]
            chunk_metas = [getattr(c, "metadata", {}) for c in chunks]

            # Basic validation
            nonempty_texts = [t for t in chunk_texts if t and t.strip()]
            if not nonempty_texts:
                raise ValueError("All chunk texts are empty after splitting. Check text extraction and splitter settings.")

            if len(chunk_texts) != len(chunk_metas):
                raise ValueError("Mismatch between number of chunk_texts and chunk_metas")

            # Create/load FAISS via manager
            fm = FaissManager(self.faiss_dir, self.model_loader)

            try:
                vs = fm.load_or_create(texts=chunk_texts, metadatas=chunk_metas)
                log.info("FaissManager.load_or_create succeeded", index_dir=str(self.faiss_dir))
            except Exception as e:
                log.error("FaissManager.load_or_create failed", error=str(e))
                # As a fallback, try to create from a smaller sample to isolate the problem
                try:
                    sample_texts = chunk_texts[:10]
                    sample_metas = chunk_metas[:10]
                    log.info("Attempting fallback create with first 10 chunks")
                    vs = fm.load_or_create(texts=sample_texts, metadatas=sample_metas)
                    # if fallback succeeded, add remaining docs
                    remaining_docs = chunks[10:]
                    if remaining_docs:
                        added = fm.add_documents(remaining_docs)
                        log.info("Added remaining docs after fallback create", added=added)
                    vs = fm.get_vectorstore()
                except Exception as e2:
                    log.error("Fallback create also failed", error=str(e2))
                    raise DocumentPortalException("Failed to create or load FAISS index", e2) from e2

            # If FaissManager expects separate add_documents semantics, call it (idempotent if load_or_create already stored them)
            try:
                added = fm.add_documents(chunks)
                log.info("FaissManager.add_documents returned", added=added)
            except Exception as e:
                # Non-fatal: log and continue since load_or_create may have already written entries
                log.warning("FaissManager.add_documents failed (continuing)", error=str(e))

            # Ensure we have an up-to-date vectorstore
            vs = fm.get_vectorstore()
            if vs is None:
                raise DocumentPortalException("Vectorstore is None after creation", "vs=None")

            # Prepare retriever
            search_kwargs = {"k": k}
            if search_type == "mmr":
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR search", k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

            retriever = vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

            return retriever

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e


    
    # def build_retriver(self,
    #                    uploaded_files,
    #                    chunk_size= 250, 
    #                    chunk_overlap= 50,
    #                    k=5,
    #                    search_type= 'mmr',
    #                    fetch_k= 20,
    #                    lambda_mult=0.5):
    #     try:
    #         paths= save_uploaded_files(uploaded_files, self.temp_dir)
    #         docs= load_documents(paths)
    #         if not docs:
    #             raise ValueError("No valid documents loaded")

    #         chunks= self._split(docs,  chunk_size= chunk_size, chunk_overlap= chunk_overlap)

    #         fm= FaissManager(self.faiss_dir, self.model_loader)

    #         texts= [c.page_content for c in docs]
    #         metadata= [c.metadata for c in docs]
    #         #print(">>>>> metadata>>>", metadata)
    #         try:
    #             vs= fm.load_or_create(texts= texts, metadatas= metadata)
    #         except Exception:
    #             vs= fm.load_or_create(texts= texts, metadatas= metadata)

    #         added= fm.add_documents(chunks)
    #         log.info("FAISS index updated", added= added, index= str(self.faiss_dir))

    #         search_kwargs= {"k": k}

    #         if search_type == "mmr":
    #             search_kwargs["fetch_k"]= fetch_k
    #             search_kwargs["lambda_mult"]= lambda_mult
    #             log.info("Using MMR search", k=k, fetch_k= fetch_k, lambda_mult=lambda_mult)

    #         return vs.as_retriever(search_type= search_type, search_kwargs= search_kwargs)
        
    #     except Exception as e:
    #         log.error("Failed to build retriever", error= str(e))
    #         raise DocumentPortalException("Failed to build retiever", e) from e
