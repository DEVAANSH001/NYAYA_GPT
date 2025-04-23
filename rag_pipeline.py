from langchain_google_genai import ChatGoogleGenerativeAI
from vector_database import faiss_db, load_pdf_to_faiss 
import re
import os
from dotenv import load_dotenv
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

HARM_CATEGORY_DANGEROUS = 1  
BLOCK_NONE = 0  

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    max_output_tokens=512,
    safety_settings={HARM_CATEGORY_DANGEROUS: BLOCK_NONE},
)

def retrieve_docs(query):
    try:
        if not faiss_db:
            raise ValueError("FAISS database is not initialized.")
        docs = faiss_db.similarity_search(query, k=3)
        return docs if docs else []
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []
    
    
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Improved text splitting
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,   
        chunk_overlap=200,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""], 
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def get_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)

def clean_output(text):
    
    text = re.sub(r"(?i)\b(?:might|could|possibly|perhaps)\b", "", text)
    text = re.sub(r"\bSection\s+\d+\b(?! of)", "", text)  
    
    text = re.sub(r"\n{3,}", "\n\n", text)  
    return text.strip()


def hybrid_retrieval(query):
    try:
        if not faiss_db:
            raise ValueError("FAISS database is not initialized.")
        
        
        keyword_docs = faiss_db.similarity_search(query, k=2)
        
        
        mmr_docs = faiss_db.max_marginal_relevance_search(query, k=3)
        
        
        combined = keyword_docs + mmr_docs
        seen = set()
        unique_docs = []
        for doc in combined:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        return unique_docs[:4]  
    except Exception as e:
        print(f"Error during hybrid retrieval: {e}")
        return []


def answer_query(documents, model, query):
    if not documents or isinstance(documents[0], str):
        return "No relevant legal documents found."
    
    context = get_context(documents)
    print(f"Retrieved Context: {context}")  

    prompt = f"""
**Legal Analysis Task**
You are NyayaGPT, an expert legal assistant specialized in Indian law. Analyze the context and question carefully.

**Context Analysis Guidelines:**
1. Identify key legal provisions, sections, and case laws
2. Note any definitions, exceptions, or limitations
3. Highlight relevant precedents or judicial interpretations

**Response Requirements:**
- Structure your answer with clear headings
- Cite exact legal provisions when available
- Use bullet points for multiple elements
- If context is insufficient, state what's missing
- NEVER invent laws or provisions

**Context:**
{context}

**Question:**
{query}

**Legal Analysis:**
"""
    response = model.invoke(prompt)
    return clean_output(response.content)
    