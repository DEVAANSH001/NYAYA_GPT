from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
import os

pdfs_directory = 'pdfs/'
FAISS_DB_PATH = "vectorstore/db_faiss"

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def load_pdf_to_faiss(file_path):
    documents = load_pdf(file_path)
    chunks = create_chunks(documents)
    faiss_db = FAISS.from_documents(chunks, get_embedding_model())
    faiss_db.save_local(FAISS_DB_PATH)
    return faiss_db

if os.path.exists(FAISS_DB_PATH):
    faiss_db = FAISS.load_local(FAISS_DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)
else:
    faiss_db = None
