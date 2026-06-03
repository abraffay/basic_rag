from langchain_core.prompts import ChatPromptTemplate

from .loader import (
    load_documents,
    split_documents
)

from .embeddings import (
    get_embeddings
)

from .retriever import (
    HybridRetriever
)

from .reranker import (
    Reranker
)

from .llm import (
    get_llm
)


documents = load_documents()

chunks = split_documents(
    documents
)

embeddings = get_embeddings()

retriever = HybridRetriever(
    chunks,
    embeddings
)

reranker = Reranker()

llm = get_llm()


prompt = (
    ChatPromptTemplate.from_template(
        """
Use only the context below.

Context:

{context}

Question:

{question}

Answer:
"""
    )
)


def format_docs(docs):

    return "\n\n".join(
        f"[{d.metadata.get('source')}] "
        f"{d.page_content}"
        for d in docs
    )


def answer_question(query):

    docs = retriever.retrieve(
        query
    )

    docs = reranker.rerank(
        query,
        docs,
        top_k=5
    )
    print('reranked docs', docs)
    print("--------------------------------")

    context = format_docs(
        docs
    )

    chain = (
        prompt
        | llm
    )

    return chain.invoke(
        {
            "context": context,
            "question": query
        }
    )