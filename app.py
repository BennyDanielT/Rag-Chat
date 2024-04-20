import streamlit as st
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from components.conversation_chain import QuestionAnswering
from utils.load_config import LoadConfig
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

QA = QuestionAnswering()
CFG = LoadConfig()

load_dotenv()


def get_pdf_text(pdf_docs):
    raw_text = ""
    for pdf_doc in pdf_docs:
        # with open(pdf_doc.name, mode="wb") as w:
        #     w.write(pdf_doc.getvalue())
        # loader = PyPDFLoader(pdf_doc.name)
        # docs = loader.load_and_split()
        loader = PdfReader(pdf_doc)
        for page in loader.pages:
            raw_text += "\n\n".join(page.extract_text())
        # print(raw_text)
    return raw_text


def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.chunk_size, chunk_overlap=CFG.chunk_overlap
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def get_retriever(text_chunks):

    # vectorstore = Chroma.from_texts(texts=text_chunks, embedding=OpenAIEmbeddings())
    vectorstore = FAISS.from_texts(text_chunks, embedding=OpenAIEmbeddings())
    vectorstore.save_local("faiss_index")

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    return retriever


def get_rag_chain(pdf_files):
    text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(text)
    retriever = get_retriever(text_chunks)
    prompt = QA.get_prompt()
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def generate_answers(rag_chain, query):
    # text = get_pdf_text(pdf_files)
    # text_chunks = get_text_chunks(text)
    # vectorstore = get_retriever(text_chunks)
    # prompt = QA.get_prompt()
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    # rag_chain = get_rag_chain(prompt=prompt, llm=llm, retriever=vectorstore)
    response = rag_chain.invoke(query)
    return response


st.set_page_config(
    layout="wide", page_title="Interact with Documents", page_icon=":computer:"
)
st.title("Start chatting now")

st.sidebar.title("Settings")
# st.title("Home:")
pdf_docs = st.file_uploader(
    "Upload your documents here",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)
if st.button("Upload"):
    st.spinner("Processing documents...")
if pdf_docs:
    chain = get_rag_chain(pdf_docs)

    user_question = st.text_input("Enter your query here...")
    if user_question:
        response = generate_answers(chain, user_question)
        st.write(response)
