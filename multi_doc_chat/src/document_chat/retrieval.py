import os
import sys
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger.custom_logger import CustomLogger
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.models.models import PromptType, ChatAnswer
from pydantic import ValidationError

log = CustomLogger().get_logger(__name__)

class ConversationalRag:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id, retriever= None):

        try:
            self.session_id= session_id

            self.llm= self._load_llm()
            self.contextualize_prompt= PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt= PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]
            self.retriever= retriever
            self.chain= None
            if self.retriever is not None:
                self._build_lcel_chain()
            log.info("ConversationalRag Initialized", session_id= self.session_id)

        except Exception as e:
            log.error("Failed to initialize ConversationalRag", error= str(e))
            raise DocumentPortalException("Failed to initialize ConversationalRag", sys)
        
    def load_retriever_from_faiss(self,
                                  index_path,
                                  k=5,
                                  index_name= 'index',
                                  search_type= 'mmr',
                                  fetch_k= 20,
                                  lambda_mult= 0.5,
                                  search_kwargs= None):
        
        try:
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS directroy not found : {index_path}")
            embeddings= ModelLoader().load_embedding()
            vector_store= FAISS.load_local(
                index_path,
                embeddings, 
                index_name= index_name,
                allow_dangerous_deserialization= True
            )
            if search_kwargs is None:
                search_kwargs = {"k": k}
                if search_type == "mmr":
                    search_kwargs["fetch_k"] = fetch_k
                    search_kwargs["lambda_mult"] = lambda_mult
                    
            self.retriever= vector_store.as_retriever(
                search_type= search_type, search_kwargs= search_kwargs
            )
            self._build_lcel_chain()

            log.info("FAISS retriever loaded succesfully",
                     index_path= index_path,
                     index_name= index_name,
                     search_type= search_type,
                     k= k,
                     fetch_k= fetch_k if search_type == 'mmr' else None,
                     lambda_mult= lambda_mult if search_type == 'mmr' else None,
                     session_id= self.session_id)
            return self.retriever
        
        except Exception as e:
            log.error("Failed to load retriver from FAISS", error= str(e))
            raise DocumentPortalException("Failed to load retriver from FAISS", sys)
        

    
        
    def invoke(self, user_input, chat_history= None):
        try:
            if self.chain is None:
                raise DocumentPortalException("RAG chain is not initialized. Exceute load_retriever_from_faiss() before invoke()", sys)
            
            chat_history= chat_history or []

            ##### Retrieval Block
            # rewritten_query = user_input  # If you want, you can add rewriting logic here too
            # retrieved_docs = self.retriever.get_relevant_documents(rewritten_query)

            # print("\n🔍 Retrieved Documents:")
            # for i, doc in enumerate(retrieved_docs, start=1):
            #     print(f"\n--- Document {i} ---")
            #     print(doc.page_content[:500])  # Print first 500 chars for readability
            #     print("----------------------")

            ######

            payload= {"input": user_input, "chat_history": chat_history}
            answer= self.chain.invoke(payload)

            if not answer:
                log.warning("No answer generated", user_input= user_input, session_id= self.session_id)
                return "No answer generated"
            
            try:
                validated= ChatAnswer(answer= str(answer))
                answer= validated.answer
            except ValidationError as ve:
                log.error("Invalid Answer", error= str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)
            log.info("Chain invoked sucesfully",
                     session_id= self.session_id,
                     user_input= user_input)
            return answer
        except Exception as e:
            log.error("Failed to invoke chain ConversationalRag", error= str(e))
            raise DocumentPortalException("Failed to invoke chain ConversationalRag", sys)
        
    def _load_llm(self):
        try:
            llm= ModelLoader().load_llm()
            if not llm:
                raise ValueError("Failed to load LLM")
            log.info("LLM loaded succesfully", sesson_id= self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error= str(e))
            raise DocumentPortalException("Failed to load LLM in ConversationalRag", sys)
        
    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                print(self.retriever)
                raise DocumentPortalException("No retriever is available befor building LCEL", sys)
            
            #Rewrite user question with chat history context
            question_rewritter= (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )   

            #Retrieve docs for rewritten query
            retrieved_docs= question_rewritter | self.retriever | self._format_docs

            #Answer using retrived docs + original input + chat history
            self.chain= (
                {
                    "context": retrieved_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history")
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )   
            log.info("LCEL built succesfully", session_id= self.session_id)

        except Exception as e:
            log.error("Failed to build LCEL", error= str(e), session_id= self.session_id)
            raise DocumentPortalException("Failed to build LCEL", sys)      