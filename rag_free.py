from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
model = SentenceTransformer("all-MiniLM-L6-v2")
documents = [
    "AI is used in healthcare for diagnosis and treatment.",
    "Machine learning helps systems learn from data.",
    "Deep learning is a subset of machine learning.",
    "Cricket is very popular in Pakistan.",
    "Football is a global sport played worldwide."
]

chunks = documents
vectors = model.encode(chunks)
vectors = np.array(vectors).astype("float32")


dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(vectors)

def retrieve(query, k=2):
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    return [chunks[i] for i in indices[0]]


def generate_answer(query, context):
    return f"""
Question: {query}

Answer based on knowledge:
{context}

(Simulated LLM response)
"""


while True:
    query = input("\nAsk something: ")

    results = retrieve(query)

    context = "\n".join(results)

    answer = generate_answer(query, context)

    print("\n--- RAG OUTPUT ---")
    print(answer)