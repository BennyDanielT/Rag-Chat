from abc import ABC, abstractmethod
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFDirectoryLoader
from utils.load_config import LoadConfig
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

CFG = LoadConfig()


class Document:
    def read_files(self, pdf_files) -> str:
        pdf_content = ""
        for document in pdf_files:
            file_reader = PdfReader(document)
            for page in file_reader.pages:
                pdf_content += page.extract_text()
        return pdf_content

    def read_files_from_directory(self, directory):
        file_loader = PyPDFDirectoryLoader(directory)
        documents = (
            file_loader.load()
        )  # Returns an object of type - langchain_core.documents.base.Document
        file_contents = self.read_files(documents)
        return file_contents


class Chunk(ABC):

    @abstractmethod
    def chunk_data(self, text):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CFG.chunk_size,
            chunk_overlap=CFG.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        pass


class DataChunker(Chunk):
    def chunk_text(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks

    def chunk_documents(self, documents):
        chunks = self.text_splitter.split_documents(documents)
        return chunks
