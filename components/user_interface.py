import streamlit as st
from components.document_operations import Document, DataChunker
from components.vector_store_functions import CHROMA
from components.conversation_chain import QuestionAnswering
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI


class UserInterface:

    def __init__(self):
        # self.DOC = Document()
        # self.CHUNKER = DataChunker()
        # self.VECTOR_STORE = CHROMA()
        self.llm = ChatMistralAI(
            model="mistral-large-latest"
        )  # TODO: Update LLM in Config
        st.set_page_config(
            layout="wide", page_title="Interact with Documents", page_icon=":computer:"
        )

        st.title("Start chatting now")
        self.query = st.text_input("Enter your query here")
        with st.sidebar:
            st.title("Home:")
            self.docs = st.file_uploader(
                "Upload your documents here",
                type=["pdf", "docx"],
                accept_multiple_files=True,
            )
            QA = QuestionAnswering(documents = self.docs)
            self.rag_chain = QA.get_rag_chain()

            if(self.query):
                self.rag_chain.invoke(self.query)


            if st.button("Upload"):
                with st.spinner("Processing documents..."):
                    # self.raw_content = self.DOC.read_documents(pdf_files=self.docs)
                    # self.chunks = self.CHUNKER.chunk_data(self.raw_content)
                    # self.vectordb = self.VECTOR_STORE.generate_vectordb(
                    #     chunks=self.chunks
                    # )  # TODO: Fix issue with long processing time for large files
                    # self.retriever = self.vectordb.as_retriever()
                    # self.prompt = self.QA.get_prompt()
                    # self.rag_chain = (
                    #     {
                    #         "context": self.retriever,
                    #         "question": RunnablePassthrough(),
                    #     }
                    #     | self.prompt
                    #     | self.llm
                    #     | StrOutputParser()
                    # )

                    st.success("Documents uploaded successfully!")

        if self.query and self.rag_chain:
            self.rag_chain.invoke(self.query)
