import streamlit as st
import re
from rag_pipeline import answer_query, retrieve_docs, llm_model

st.markdown(
    """
    <style>
    /* Universal text color for better visibility */
    .chat-container, .user-message, .ai-message {
        color: #000000;
    }

    @media (prefers-color-scheme: dark) {
        .chat-container {
            background-color: #1e1e1e;
            border: 1px solid #444;
        }
        .user-message {
            background-color: #2e7d32;
            color: #ffffff;
        }
        .ai-message {
            background-color: #333333;
            color: #ffffff;
        }
    }

    @media (prefers-color-scheme: light) {
        .chat-container {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
        }
        .user-message {
            background-color: #DCF8C6;
            color: #000000;
        }
        .ai-message {
            background-color: #F1F0F0;
            color: #000000;
        }
    }

    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 10px;
        max-height: 500px;
        overflow-y: auto;
        direction: rtl;
    }
    .chat-container > div {
        direction: ltr;
    }
    .user-message, .ai-message {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)
    st.header("Instructions")
    st.markdown(
        """
        - **Upload a PDF:** Upload the Declaration PDF (optional).
        - **Enter your prompt:** Ask legal-related queries.
        - **Chat:** Engage in a legal conversation.
        - **Theme:** Auto-adjusts to dark or light mode.
        """
    )

st.title("NyayaGPT")

with st.form(key="chat_form", clear_on_submit=True):
    user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask Anything!")
    submit_button = st.form_submit_button("Send")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def parse_response(response_text: str):
    """ Extracts only the final answer, ignoring chain-of-thought. """
    final_answer = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    return final_answer

if submit_button:
    if not user_query.strip():
        st.error("Please enter your query before sending!")
    else:
        st.chat_message("user").write(user_query)

        
        if uploaded_file:
            retrieved_docs = retrieve_docs(user_query)
            response_text = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        else:
           
            prompt = f"""
You are NyayaGPT, an expert legal assistant. Answer the question below accurately, and explain in bullet points where needed.
If the context is unrelated to law, respond with:
 "As a legal specialist, I can only provide information based on applicable legal content. The provided context does not appear to relate to law."

Question:
{user_query}

Answer:
"""
            response = llm_model.invoke(prompt)
            response_text = response.content

        final_answer = parse_response(response_text)


        st.session_state.chat_history.append({"role": "user", "message": user_query})
        st.session_state.chat_history.append({"role": "NyayaGPT", "message": final_answer})


st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>You:</strong><br>{msg["message"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai-message"><strong>NyayaGPT:</strong><br>{msg["message"]}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
