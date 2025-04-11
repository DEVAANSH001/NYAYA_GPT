# from langchain_groq import ChatGroq
# from vector_database import faiss_db
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv


# # Load environment variables
# load_dotenv()

# # ✅ Step 1: Setup LLM
# llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")


# #Step2: Retrieve Docs

# def retrieve_docs(query):
#     return faiss_db.similarity_search(query)

# def get_context(documents):
#     context = "\n\n".join([doc.page_content for doc in documents])
#     return context

# #Step3: Answer Question

# custom_prompt_template = """
# Use the pieces of information provided in the context to answer user's question.
# If you dont know the answer, just say that you dont know, dont try to make up an answer. 
# Dont provide anything out of the given context
# Question: {question} 
# Context: {context} 
# Answer:
# """

# # def answer_query(documents, model, query):
# #     context = get_context(documents)
# #     prompt = ChatPromptTemplate.from_template(custom_prompt_template)
# #     chain = prompt | model
# #     return chain.invoke({"question": query, "context": context})


# def answer_query(documents, model, query):
#     context = get_context(documents)
#     prompt = ChatPromptTemplate.from_template(custom_prompt_template)
#     chain = prompt | model
#     output = chain.invoke({"question": query, "context": context})
#     # Since output is an AIMessage, use its content attribute
#     answer_text = output.content
#     return answer_text


# #question="If a government forbids the right to assemble peacefully which articles are violated and why?"
# #retrieved_docs=retrieve_docs(question)
# #print("AI Lawyer: ",answer_query(documents=retrieved_docs, model=llm_model, query=question))
from langchain.llms import HuggingFaceHub
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate
import re
import streamlit as st
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# ✅ Step 1: Load API Key from Streamlit Secrets

# ✅ Step 2: Initialize LLM

llm_model = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2"  # Try the latest version
,  # Choose any hosted model
    model_kwargs={"temperature": 0.7, "max_length": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
)

# ✅ Step 3: Retrieve Docs
def retrieve_docs(query):
    retrieved_documents = faiss_db.similarity_search(query)
    print("Retrieved Documents:", retrieved_documents)  # Debugging output
    if not retrieved_documents:
        print("⚠️ No documents retrieved for query:", query)
    return retrieved_documents


def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# ✅ Step 4: Answer Query
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know—do not make up an answer.
Answer only from the context given, if a question is asked and there is no relevant context say that you don't know.
Only provide the direct answer. Do NOT repeat the question or context.
Do NOT add extra explanations or introductory phrases.

Context:
{context}

Question:
{question}

Answer:
"""





def clean_output(output):
    # Remove any labels like "Question:", "Context:", etc.
    output = re.sub(r'(?i)(?:question:|context:)', '', output).strip()

    # Ensure it doesn't contain the full prompt accidentally
    if "Answer:" in output:
        output = output.split("Answer:")[-1].strip()

    return output


def answer_query(documents, model, query):
    if not documents:
        return "No relevant documents found for this query."

    context = get_context(documents)

    # Generate prompt
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    output = chain.invoke({"question": query, "context": context})

    if not output:
        return "No response from the model."

    return clean_output(output.content if hasattr(output, "content") else str(output))
