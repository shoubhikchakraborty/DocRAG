import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor
from multi_doc_chat.src.document_chat.retrieval import ConversationalRag
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

# keep existing behavior if user didn't set this env
os.environ.setdefault("LANGSMITH_TRACING", os.getenv("LANGSMITH_TRACING", "true"))

def test_RAG():
    try:
        test_files = [
            r"C:\Users\chakr\Downloads\projects\deploy_rag\data\1706.03762v7.pdf"
        ]

        uploaded_files = []

        for file_path in test_files:
            p = Path(file_path)
            if p.exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No files uploaded")
            sys.exit(1)

        # Create ingestor (this will create a session dir)
        ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)

        # Build retriever from uploaded files (this will index chunk-level texts)
        retriever = ci.build_retriver(
            uploaded_files,
            chunk_size=200,
            chunk_overlap=50,
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Close file handles
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        # Quick local retrieval sanity checks BEFORE wiring into the RAG chain
        try:
            debug_docs = retriever.get_relevant_documents("What is this paper about?")
            print(f"\n[DEBUG] retriever.get_relevant_documents returned {len(debug_docs)} docs")
            if debug_docs:
                print("[DEBUG] First doc sample:\n", debug_docs[0].page_content[:400])
        except Exception as e:
            print("[DEBUG] Error calling retriever.get_relevant_documents:", str(e))

        # Pass the retriever directly to ConversationalRag to avoid re-loading from disk
        rag = ConversationalRag(session_id=ci.session_id, retriever=retriever)

        chat_history = []
        print("\nType 'exit' to quit the chat")
        while True:
            try:
                user_input = input("You:").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat")
                break

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q", ":q"]:
                print("Good Bye !!!")
                break

            try:
                answer = rag.invoke(user_input, chat_history=chat_history)
            except Exception as e:
                print("Assistant: (error invoking RAG) ", str(e))
                # record the failed exchange in history for troubleshooting if desired
                chat_history.append(HumanMessage(content=user_input))
                chat_history.append(AIMessage(content=f"(error) {str(e)}"))
                continue

            print("Assistant: ", answer)

            # update chat history with LangChain message types
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        

    except Exception as e:
        print("Test script failed:", str(e))
        sys.exit(1)


if __name__ == "__main__":
    test_RAG()
