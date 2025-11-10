import os
import json
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from multi_doc_chat.logger.custom_logger import CustomLogger 
from multi_doc_chat.exceptions.custom_exception import DocumentPortalException
from multi_doc_chat.utils.config_loader import load_config

log = CustomLogger().get_logger(__name__)

class ApiKeyManager:
    REQUIRED_KEYS= ['GROQ_API_KEY']

    def __init__(self):
        self.api_keys= {}
        raw= os.getenv("API_KEYS")
        
        if raw:
            try:
                parsed= json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API Keys is not a valid JSON object")
                self.api_keys= parsed
                log.info("Loaded API KEYS")
            except Exception as e:
                log.warning("Failed to parse API KEYS as JSON", error= str(e))
        
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val= os.getenv(key)
                if env_val:
                    self.api_keys[key]= env_val
                    log.info(f"Loaded {key} from individual env var")
        
        missing= [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.info("API KEYS missing")
            raise DocumentPortalException("Missing API Keys", sys)
        
        log.info("API Keys loaded", keys= {k : v[:6] + "....." for k, v in self.api_keys.items()})
        

    def get(self, key:str):
        val= self.api_keys.get(key)
        if not val:
            raise KeyError("API key {key} is missing")
        return val
    

class ModelLoader:

    def __init__(self):
        if os.getenv("ENV", "LOCAL").lower() != 'production':
            load_dotenv()
            log.info("Running in local env")

        else:
            log.info("Running in Prod env")

        self.api_key_mgr= ApiKeyManager()
        self.config= load_config()
        log.info("Config YAML loaded", config_keys= list(self.config.keys()))

    def load_embedding(self):

        try:
            model_name= self.config['embedding_model']['model_name']
            return HuggingFaceEmbeddings(model_name= model_name)
        
        except Exception as e:
            log.error("Error loading embedding model", error= str(e))
            raise DocumentPortalException("Failed to embedding model", sys)
        
    def load_llm(self):
        llm_config_block= self.config['llm']
        llm_provider= os.getenv("LLM_PROVIDER", "groq")

        if llm_provider not in llm_config_block:
            log.error("LLM provider not found in config", provider= llm_provider)
            raise ValueError(f"LLM provider {llm_provider} not found in config")
        
        llm_config= llm_config_block[llm_provider]
        provider= llm_config['provider']
        model_name= llm_config['model_name']
        temperature= llm_config['temperature']
        max_output_tokens= llm_config['max_output_tokens']

        if provider == 'groq':
            return ChatGroq(model= model_name,
                            api_key= self.api_key_mgr.get("GROQ_API_KEY"),
                            temperature= temperature
                            )
        else:
            log.error("Unsupported LLM provider", provider= llm_provider)
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        

if __name__ == "__main__":

    loader= ModelLoader()

    embeddings= loader.load_embedding()
    print(f"Embedding Model Loaded: {embeddings}")
    result= embeddings.embed_query("How are you")
    print(f"Embedding Result: {len(result)}")

    llm= loader.load_llm()
    print(f"LLM loaded: {llm}")
    result= llm.invoke("Hello, how are you?")
    print(f"LLM Result : {result.content}")