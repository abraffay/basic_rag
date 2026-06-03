from pathlib import Path
import numpy as np

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


# ======================================
# LOAD PDFs
# ======================================

docs_folder = Path("docs")

all_docs = []

for file in docs_folder.glob("*.pdf"):

    loader = PyMuPDFLoader(str(file))

    docs = loader.load()

    for d in docs:
        d.metadata["source"] = file.name

    all_docs.extend(docs)

print("Pages:", len(all_docs))


# ======================================
# CHUNKING
# ======================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(all_docs)


# ======================================
# TEXT FOR BM25
# ======================================

tokenized_chunks = [
    doc.page_content.lower().split()
    for doc in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


# ======================================
# EMBEDDINGS + VECTOR DB
# ======================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embeddings
)

vector_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 30
    }
)


# ======================================
# RERANKER
# ======================================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank(query, docs, top_k=5):

    pairs = [
        (query, d.page_content)
        for d in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [d for d, _ in ranked[:top_k]]


# ======================================
# HYBRID RETRIEVAL
# ======================================

def hybrid_retrieve(query):

    query_tokens = query.lower().split()

    # BM25 results
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_top_idx = np.argsort(bm25_scores)[::-1][:10]

    bm25_docs = [chunks[i] for i in bm25_top_idx]

    # Vector results
    vector_docs = vector_retriever.invoke(query)

    # Merge
    combined = bm25_docs + vector_docs

    # Remove duplicates
    seen = set()
    unique_docs = []

    for d in combined:
        if d.page_content not in seen:
            unique_docs.append(d)
            seen.add(d.page_content)

    # Rerank
    return rerank(query, unique_docs, top_k=5)


# ======================================
# LLM
# ======================================

llm = Ollama(model="llama3")


prompt = ChatPromptTemplate.from_template(
"""
Use only the context below.

Context:
{context}

Question:
{question}

Answer clearly.
"""
)


def format_docs(docs):

    return "\n\n".join(
        f"[{d.metadata.get('source')}] {d.page_content}"
        for d in docs
    )


# ======================================
# FINAL PIPELINE
# ======================================

def answer(query):

    docs = hybrid_retrieve(query)

    context = format_docs(docs)

    chain = (
        {
            "context": RunnableLambda(lambda _: context),
            "question": RunnableLambda(lambda _: query)
        }
        | prompt
        | llm
    )

    return chain.invoke({})


# ======================================
# CHAT LOOP
# ======================================

while True:

    q = input("\nAsk: ")

    if q.lower() == "exit":
        break

    print("\n" + "=" * 60)
    print(answer(q))
    print("=" * 60)