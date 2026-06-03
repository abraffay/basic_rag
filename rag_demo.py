from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("all-MiniLM-L6-v2")
docs = [
    "AI is used in healthcare",
    "Machine learning helps predictions",
    "Cricket is popular in Pakistan",
    "Football is a global sport"
]

vectors = model.encode(docs)
print('vectors: ', vectors)
vectors = np.array(vectors).astype("float32")
print('vectors: ', vectors)

dimension = vectors.shape[1]
print('dimension: ', dimension)
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

def search(query, k=2):
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    return [docs[i] for i in indices[0]]


print(search("AI in medicine"))