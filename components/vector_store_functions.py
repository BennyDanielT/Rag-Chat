from abc import ABC, abstractmethod
from langchain.vectorstores import FAISS
from utils.load_config import LoadConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from langchain_chroma import Chroma

CFG = LoadConfig()
CFG.load_model()


class VectorStore(ABC):
    @abstractmethod
    def generate_vectordb(self, text):
        pass


class FAISS(VectorStore):
    def generate_vectordb(self, documents):
        vectordb = FAISS.from_texts(
            docs=documents, embedding=CFG.embedding_mode
        )  # TODO: Update Embedding model in the config file
        vectordb.save_local(CFG.vectordb_directory)
        return vectordb


class CHROMA(VectorStore):
    def generate_vectordb(self, chunks):
        vectordb = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
        return vectordb
