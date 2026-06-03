from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

query = "How does transfer learning work?"

docs = [
    "Neural networks consist of layers.",
    "Transfer learning reuses pretrained models.",
    "Image augmentation improves datasets.",
    "CNNs are popular for computer vision."
]


pairs = [
    (query, doc)
    for doc in docs
]

scores = reranker.predict(pairs)

ranked = sorted(
    zip(docs, scores),
    key=lambda x: x[1],
    reverse=True
)

for doc, score in ranked:
    print(score)
    print(doc)