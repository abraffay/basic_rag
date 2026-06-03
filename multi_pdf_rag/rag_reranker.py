from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from langchain_community.llms import Ollama


# ==========================================
# LOAD PDFS
# ==========================================

all_docs = []

docs_folder = Path("docs")

for pdf_file in docs_folder.glob("*.pdf"):

    loader = PyMuPDFLoader(str(pdf_file))

    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = pdf_file.name

    all_docs.extend(docs)

print(f"Loaded {len(all_docs)} pages")


# ==========================================
# CHUNKING
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_documents(all_docs)

print(f"Created {len(chunks)} chunks")


# ==========================================
# EMBEDDINGS
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# VECTOR DB
# ==========================================

db = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "fetch_k": 50
    }
)


# ==========================================
# RERANKER
# ==========================================

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank_docs(
    query,
    docs,
    top_k=5
):
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked[:top_k]
    ]


# ==========================================
# FORMAT DOCS
# ==========================================

def format_docs(docs):

    result = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        result.append(
            f"[SOURCE: {source}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(result)


# ==========================================
# LLM
# ==========================================

llm = Ollama(
    model="llama3"
)


# ==========================================
# PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Use ONLY the context below.

If the answer is not in the context,
say you do not know.

Context:

{context}

Question:

{question}

Answer:
"""
)


# ==========================================
# RAG PIPELINE
# ==========================================

def answer_question(question):

    docs = retriever.invoke(question)

    docs = rerank_docs(
        question,
        docs,
        top_k=5
    )

    context = format_docs(docs)

    chain = (
        {
            "context": RunnableLambda(
                lambda _: context
            ),
            "question": RunnableLambda(
                lambda _: question
            )
        }
        | prompt
        | llm
    )

    return chain.invoke({})


# ==========================================
# CHAT LOOP
# ==========================================

while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    answer = answer_question(query)

    print("\n" + "=" * 50)
    print(answer)
    print("=" * 50)