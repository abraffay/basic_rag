import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

def build_index(chunks):
    vectors = model.encode(chunks)
    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(vectors)

    return index, vectors

def search(query, chunks, index, k=3):
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    return [chunks[i] for i in indices[0]]



def ask_llm(question, context):
    prompt = f"""
You are a helpful assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and simply.
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

model = SentenceTransformer("all-MiniLM-L6-v2")


pdf_text = load_pdf("data.pdf")
chunks = chunk_text(pdf_text)

index, vectors = build_index(chunks)

while True:
    query = input("\nAsk your PDF: ")

    relevant_chunks = search(query, chunks, index)

    context = "\n".join(relevant_chunks)

    answer = ask_llm(query, context)

    print("\n--- ANSWER ---\n")
    print(answer)