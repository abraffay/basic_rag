import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


documents = [
    "AI is used in healthcare for diagnosis and treatment.",
    "Machine learning helps systems learn from data.",
    "Deep learning is a subset of machine learning.",
    "Cricket is very popular in Pakistan.",
    "Football is a global sport played worldwide.",
    "Ronaldo is GOAT of football.",
    "Imran Khan is the best cricket player of all time"
]


vectors = model.encode(documents)
vectors = np.array(vectors).astype("float32")

dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

def retrieve(query, k=2):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [documents[i] for i in indices[0]]

def ask_llm(question, context):
    prompt = f"""
You are a helpful AI assistant.

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

    print('response: ', response.json())
    return response.json()["response"]




while True:
    query = input("\nAsk: ")

    docs = retrieve(query)

    context = "\n".join(docs)

    answer = ask_llm(query, context)

    print("\n--- ANSWER ---\n")
    print(answer)