from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


docs_path = Path("docs")

all_docs = []

for pdf_file in docs_path.glob("*.pdf"):

    loader = PyMuPDFLoader(
        str(pdf_file)
    )

    docs = loader.load()

    for d in docs:
        d.metadata["source_file"] = pdf_file.name

    all_docs.extend(docs)

print(
    "Loaded docs:",
    len(all_docs)
)



splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(
    all_docs
)
print(
    "Chunks:",
    len(chunks)
)



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.from_documents(
    chunks,
    embeddings
)


db.save_local(
    "vector_store"
)

print(
    "Database saved!"
)