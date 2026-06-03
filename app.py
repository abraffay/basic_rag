import streamlit as st
import fitz
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def build_index(chunks):
    vectors = model.encode(chunks)
    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index

def search(query, chunks, index, k=3):
    q_vec = model.encode([query]).astype("float32")

    _, indices = index.search(q_vec, k)

    return [chunks[i] for i in indices[0]]


def ask_llm(question, context):
    prompt = f"""
You are a helpful AI assistant.

Context:
{context}

Question:
{question}

Answer clearly.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]



st.title("📄 PDF RAG Chatbot (Free AI)")
print('st.title: ')
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = load_pdf(uploaded_file)
    chunks = chunk_text(text)

    st.success("PDF loaded successfully!")

    index = build_index(chunks)

    query = st.text_input("Ask a question from your PDF")

    if query:
        results = search(query, chunks, index)
        context = "\n".join(results)

        answer = ask_llm(query, context)

        st.subheader("Answer")
        st.write(answer)