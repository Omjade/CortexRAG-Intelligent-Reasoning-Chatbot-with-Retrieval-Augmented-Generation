import os
import shutil
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Define paths
PDFS_DIRECTORY = "pdfs/"
FAISS_DB_PATH = "vectorstore/db_faiss"

# Ensure directories exist
os.makedirs(PDFS_DIRECTORY, exist_ok=True)
os.makedirs("vectorstore", exist_ok=True)

# Function to save uploaded file dynamically
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDFS_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path  # Return file path for further processing

# Function to load PDF dynamically
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    document = loader.load()
    return document

# Step 2: Create chunks dynamically
def create_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(document)

# Step 3: Setup embedding model
ollama_model_name = "deepseek-r1:1.5b"
def get_embedding_model():
    return OllamaEmbeddings(model=ollama_model_name)

# Step 4: Function to process and store PDF in FAISS dynamically
def process_and_store_pdf(uploaded_file):
    file_path = save_uploaded_file(uploaded_file)  # Save file
    document = load_pdf(file_path)  # Load PDF
    text_chunks = create_chunks(document)  # Split into chunks
    print("Chunks Count:", len(text_chunks))

    # Index new document embeddings in FAISS
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(FAISS_DB_PATH)  # Save the updated FAISS database
    print("FAISS database updated with new document.")

    return faiss_db  # Return updated FAISS instance
