import os
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()



api_key = os.getenv("GOOGLE_API_KEY")

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

pdfs_directory = './pdfs/'

# model_name = "deepseek-r1:1.5b"
# model_name = "deepseek-r1:7b"
model_name = "gemini-2.0-flash"


# embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
# embeddings = OllamaEmbeddings(model=model_name)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=api_key)

with st.sidebar:
    st.title("Model Settings")
    model_option = st.selectbox(
        "Choose a model",
        ("Google (gemini-2.0-flash)", "Ollama (deepseek-r1:1.5b)")
    )


# Update the embeddings and model based on the selected option
if model_option == "Google":
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    model = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
else:
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    model = OllamaLLM(model="deepseek-r1:1.5b")

vector_store = InMemoryVectorStore(embeddings)

def upload_pdf(file):
    # if pdf folder does not exist, create it
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)