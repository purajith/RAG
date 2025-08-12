# streamlit_app.py
import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000"  # Change if running in Docker or cloud

st.set_page_config(page_title="RAG Demo", layout="centered")

st.title("📄 RAG Pipeline - Document Q&A")

# --- 1. Upload Section ---
st.header("Upload a Document")
uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded_file is not None:
    if st.button("Upload to RAG"):
        with st.spinner("Uploading and processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{API_URL}/upload", files=files)

        if response.status_code == 200:
            st.success(f"✅ {uploaded_file.name} uploaded and retriever created!")
        else:
            st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")

# --- 2. Ask Questions Section ---
st.header("Ask a Question")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying document..."):
            payload = {"question": question}
            response = requests.post(f"{API_URL}/ask", json=payload)

        if response.status_code == 200:
            data = response.json()
            st.subheader("Answer")
            st.write(data["answer"])

        else:
            st.error(f"❌ Query failed: {response.json().get('detail', 'Unknown error')}")
