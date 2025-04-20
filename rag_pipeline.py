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
        docs = faiss_db.similarity_search(query, k=3)
        return docs if docs else []
    except Exception as e:
        return [f"Retrieval error: {e}"]

def get_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)

def clean_output(output):
    return re.sub(r'(?i)(question:|context:|answer:|\n)', '', output).strip()
def answer_query(documents, model, query):
    if not documents or isinstance(documents[0], str): 
        return "No relevant documents found."

    context = get_context(documents)
    prompt = f"""
You are NyayaGPT, an expert legal assistant. Your task is to answer the user's question in a well-formatted manner, using only the context provided below.

**Formatting Guidelines:**
- Use clear section headers if the content is categorized.
- Use bullet points (*) or sub-points (-) where appropriate.
- Use bold text for article titles or section headers.
- If the context is not related to law, respond with:
  "As a legal specialist, I can only provide information based on applicable legal content. The provided context does not appear to relate to law."

---

###  Context from PDF:
{context}

---

###  User's Question:
{query}

---

###  Answer:
"""

    response = model.invoke(prompt)
    return clean_output(response.content)
