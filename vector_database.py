from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS

# Step 1: Upload & Load raw PDF(s)
pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = 'universal_declaration_of_human_rights.pdf'
documents = load_pdf(file_path)
# print("PDF pages: ", len(documents))

# Step 2: Create Chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
# print("Chunks count: ", len(text_chunks))

# Step 3: Setup Embeddings Model (Using Hugging Face Embeddings)
def get_embedding_model():
    # Use a Hugging Face model for embeddings; this one runs locally without external quotas.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

# Step 4: Index Documents – Store embeddings in FAISS (vector store)
FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
faiss_db.save_local(FAISS_DB_PATH)
