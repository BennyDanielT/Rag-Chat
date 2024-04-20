from abc import ABC, abstractmethod
from PyPDF2 import PdfReader

# from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from utils.load_config import LoadConfig
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

CFG = LoadConfig()


class Document:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CFG.chunk_size, chunk_overlap=CFG.chunk_overlap
        )

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

    def read_documents(self, pdf_files) -> List[Document]:

        documents = []
        for pdf_file in pdf_files:
            raw_file = [pdf_file.read().decode("iso-8859-1")]
            document = self.text_splitter.create_documents(raw_file)
            # loader = PyPDFLoader(pdf_file)
            # docs = loader.load()
            documents.extend(document)
            print(f"Length of Documents: {len(documents)}")
        return documents


class Chunk(ABC):
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CFG.chunk_size,
            chunk_overlap=CFG.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    @abstractmethod
    def chunk_data(self, text):
        pass


class DataChunker(Chunk):
    def chunk_text(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks

    def chunk_data(self, documents) -> List[Document]:
        if documents:
            chunks = self.text_splitter.split_documents(documents)
            print(f"Length of Chunks: {len(chunks)}")
            return chunks
        else:
            return []
