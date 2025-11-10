from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from multi_doc_chat.models.models import UploadResponse, ChatRequest, ChatResponse
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRag
from langchain_core.messages import HumanMessage, AIMessage
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException


#Fastapi Initialization
app= FastAPI(title= "Multi-Doc-Chat", version= "0.1.0")

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base_dir= Path(__file__).resolve().parent
static_dir= Base_dir / "static"
templates_dir= Base_dir / "template"
app.mount("/static", StaticFiles(directory= str(static_dir)), name= "static")
templates= Jinja2Templates(directory= str(templates_dir))

#Captures the session so that chat history can be traced back
Sessions= {}



class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""
    def __init__(self, uf: UploadFile):
        self._uf= uf
        self.name= uf.filename or "file"

    def getbuffer(self):
        self._uf.file.seek(0) #This ensures that the entire file content is read every time this method is called.
        return self._uf.file.read()
    

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/", response_class= HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model= UploadResponse)
async def upload(files: List[UploadFile] = File (...)) ->UploadResponse:
    if not files:
        raise HTTPException(status_code= 400, detail= "No files Uploaded")
    try:
        #Wrap FastAPI files to preserve filename/ext and provide a read buffer
        wrapped_files= [FastAPIFileAdapter(f) for f in files]

        ingestor= ChatIngestor(use_session_dirs= True)
        session_id= ingestor.session_id
        ingestor.build_retriver(
            uploaded_files= wrapped_files
        )
          # Initialize empty history for this session
        Sessions[session_id] = []

        return UploadResponse(session_id= session_id, indexed= True, message= "Indexing complete with MMR")
    
    except DocumentPortalException as e:
        raise HTTPException(status_code= 500, detail= str(e))
    except Exception as e:
        raise HTTPException(status_code= 500, detail= str(e))



@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()
    if not session_id or session_id not in Sessions:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Build RAG and load retriever from persisted FAISS with MMR
        rag = ConversationalRag(session_id=session_id)
        index_path = f"faiss_index/{session_id}"
        rag.load_retriever_from_faiss(
            index_path=index_path,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Use simple in-memory history and convert to BaseMessage list
        simple = Sessions.get(session_id, [])
        lc_history = []
        for m in simple:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        answer = rag.invoke(message, chat_history=lc_history)

        # Update history
        simple.append({"role": "user", "content": message})
        simple.append({"role": "assistant", "content": answer})
        Sessions[session_id] = simple

        return ChatResponse(answer=answer)
    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    


if __name__ == "__main__":

    uvicorn.run("main:app", host= "0.0.0.0", port= int(os.getenv("PORT", "8000")), reload= True)