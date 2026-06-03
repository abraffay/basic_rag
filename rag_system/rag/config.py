from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_PATH = BASE_DIR / "docs"
VECTOR_DB_PATH = BASE_DIR / "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "BAAI/bge-reranker-base"

OLLAMA_MODEL = "llama3"