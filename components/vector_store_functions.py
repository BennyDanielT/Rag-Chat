from abc import ABC, abstractmethod
from langchain.vectorstores import FAISS
from utils.load_config import LoadConfig, load_env_variables
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai import genai

CFG = LoadConfig()
CFG.load_model()


class VectorStore(ABC):
    @abstractmethod
    def generate_vectordb(self, text):
        pass


class FAISS(VectorStore):
    def generate_vectordb(self, documents):
        vectordb = FAISS.from_texts(docs=documents, embedding=CFG.embedding_model)
        vectordb.save_local(CFG.vectordb_directory)
        return vectordb
