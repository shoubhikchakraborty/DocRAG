import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRag
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

load_dotenv()

os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))

def test_RAG():
    try:
        test_files= [
            r"C:\Users\chakr\Downloads\projects\deploy_rag\data\1706.03762v7.pdf"
        ]

        uploaded_files=[]

        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))

            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No files uploaded")
            sys.exit(1)

        ci= ChatIngestor(temp_base= "data", faiss_base= "faiss_index", use_session_dirs= True)

        retriever= ci.build_retriver(
            uploaded_files,
            chunk_size=200,
            chunk_overlap= 50,
            k= 5,
            search_type= "mmr",
            fetch_k= 20,
            lambda_mult= 0.5
        )
        print("retriever :", retriever)

        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        session_id= ci.session_id
        index_dir= os.path.join("faiss_index", session_id)

        rag= ConversationalRag(session_id= session_id)
        rag.load_retriever_from_faiss(
            index_path= index_dir,
            k=5,
            index_name= os.getenv("FAISS_INDEX_NAME", "index"),
            search_type="mmr",
            fetch_k= 20,
            lambda_mult= 0.5
        )

        chat_history= []
        print("\nType 'exit' to quit the chat")
        while True:
            try:
                user_input= input("You:").strip()
            except (EOFError, KeyboardInterrupt):
                print("Exiting chat")
                break

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q", ":q"]:
                print("Good Bye !!!")
                break

            answer= rag.invoke(user_input, chat_history= chat_history)

            print("Assistant: ", answer)

            chat_history.append(HumanMessage(content= user_input))
            chat_history.append(AIMessage(content= answer))

        if not uploaded_files:
            print("No valid files to upload")
            sys.exit(1)

    
        wait_for_all_tracers()

    except Exception as e:
        print("Test script failed", str(e))
        sys.exit(1)


if __name__ == "__main__":
    test_RAG()

