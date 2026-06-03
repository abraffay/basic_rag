from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_PATH


def load_documents():

    docs_path = Path(DOCS_PATH)

    if not docs_path.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_path}"
        )

    pdf_files = sorted(docs_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files in {docs_path}. "
            "Add one or more .pdf files before starting the API."
        )

    docs = []

    for pdf_file in pdf_files:

        loader = PyMuPDFLoader(
            str(pdf_file)
        )

        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.metadata["source"] = pdf_file.name

        docs.extend(loaded_docs)

    return docs


def split_documents(documents):

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

    return splitter.split_documents(documents)