import os 
import streamlit as st  # Importing Streamlit to create the web-based chat application

# Importing necessary components from LangChain for handling PDFs, text processing, and AI models
from langchain_community.document_loaders import PDFPlumberLoader  # Loads PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits text into manageable chunks
from langchain_core.vectorstores import InMemoryVectorStore  # Stores text embeddings in memory
from langchain_ollama import OllamaEmbeddings  # Embedding model from Ollama
from langchain_core.prompts import ChatPromptTemplate  # Defines the prompt template for AI responses
from langchain_ollama.llms import OllamaLLM  # Large language model from Ollama
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Google's Generative AI models

from dotenv import load_dotenv  # Import dotenv to load environment variables from a .env file
load_dotenv()  # Load environment variables (e.g., API keys)

# Retrieve the Google API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY")

# Define the chat prompt template for the AI model
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Define the directory where uploaded PDFs will be stored
pdfs_directory = './pdfs/'

# Define the AI model to be used (Google Gemini or DeepSeek)
# model_name = "deepseek-r1:1.5b"
# model_name = "deepseek-r1:7b"
model_name = "gemini-2.0-flash"

# Define embedding models (commented out lines show alternatives)
# embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
# embeddings = OllamaEmbeddings(model=model_name)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

# Sidebar for model selection
with st.sidebar:
    st.title("Model Settings")  # Display a title in the sidebar
    model_option = st.selectbox(
        "Choose a model",
        ("Google (gemini-2.0-flash)", "Ollama (deepseek-r1:1.5b)")  # Dropdown to select AI model
    )

# Update the embeddings and model based on the selected option
if model_option == "Google":
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    model = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
else:
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    model = OllamaLLM(model="deepseek-r1:1.5b")

# Create an in-memory vector store for storing document embeddings
vector_store = InMemoryVectorStore(embeddings)

def upload_pdf(file):
    """
    Saves the uploaded PDF file to the pdfs_directory.
    """
    # If the PDF storage folder does not exist, create it
    if not os.path.exists(pdfs_directory):
        os.makedirs(pdfs_directory)

    # Save the uploaded file to the designated folder
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    """
    Loads a PDF file and extracts text content using PDFPlumber.
    """
    loader = PDFPlumberLoader(file_path)  # Load the PDF file
    documents = loader.load()  # Extract text as documents
    return documents  # Return extracted documents

def split_text(documents):
    """
    Splits large text documents into smaller chunks for better processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum characters per chunk
        chunk_overlap=200,  # Overlapping text between chunks for context continuity
        add_start_index=True  # Adds an index to track chunk order
    )

    return text_splitter.split_documents(documents)  # Return split text chunks

def index_docs(documents):
    """
    Indexes the documents into the vector store for retrieval.
    """
    vector_store.add_documents(documents)  # Store the document embeddings

def retrieve_docs(query):
    """
    Searches the vector store for documents similar to the query.
    """
    return vector_store.similarity_search(query)  # Retrieve relevant documents

def answer_question(question, documents):
    """
    Generates an AI response based on the retrieved documents and user query.
    """
    context = "\n\n".join([doc.page_content for doc in documents])  # Combine retrieved text chunks
    prompt = ChatPromptTemplate.from_template(template)  # Create a prompt from the template
    chain = prompt | model  # Chain the prompt with the selected AI model

    return chain.invoke({"question": question, "context": context})  # Generate and return AI response

# File uploader widget in Streamlit
uploaded_file = st.file_uploader(
    "Upload PDF",  # Instructional text for users
    type="pdf",  # Restrict file type to PDFs
    accept_multiple_files=False  # Allow only one file at a time
)

# Process the uploaded PDF file if a file is provided
if uploaded_file:
    upload_pdf(uploaded_file)  # Save the uploaded file
    documents = load_pdf(pdfs_directory + uploaded_file.name)  # Extract text from the PDF
    chunked_documents = split_text(documents)  # Split the text into smaller chunks
    index_docs(chunked_documents)  # Index the document chunks for retrieval

    # Chat input field for user questions
    question = st.chat_input()

    # If a question is asked, process and respond
    if question:
        st.chat_message("user").write(question)  # Display the user's question
        related_documents = retrieve_docs(question)  # Retrieve relevant documents
        answer = answer_question(question, related_documents)  # Generate AI response
        st.chat_message("assistant").write(answer)  # Display the AI's response
