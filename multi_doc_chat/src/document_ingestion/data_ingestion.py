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
                       chunk_size= 250, 
                       chunk_overlap= 50,
                       k=5,
                       search_type= 'mmr',
                       fetch_k= 20,
                       lambda_mult=0.5):
        try:
            paths= save_uploaded_files(uploaded_files, self.temp_dir)
            docs= load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            chunks= self._split(docs,  chunk_size= chunk_size, chunk_overlap= chunk_overlap)

            fm= FaissManager(self.faiss_dir, self.model_loader)

            texts= [c.page_content for c in docs]
            metadata= [c.metadata for c in docs]
            #print(">>>>> metadata>>>", metadata)
            try:
                vs= fm.load_or_create(texts= texts, metadatas= metadata)
            except Exception:
                vs= fm.load_or_create(texts= texts, metadatas= metadata)

            added= fm.add_documents(chunks)
            log.info("FAISS index updated", added= added, index= str(self.faiss_dir))

            search_kwargs= {"k": k}

            if search_type == "mmr":
                search_kwargs["fetch_k"]= fetch_k
                search_kwargs["lambda_mult"]= lambda_mult
                log.info("Using MMR search", k=k, fetch_k= fetch_k, lambda_mult=lambda_mult)

            return vs.as_retriever(search_type= search_type, search_kwargs= search_kwargs)
        
        except Exception as e:
            log.error("Failed to build retriever", error= str(e))
            raise DocumentPortalException("Failed to build retiever", e) from e
