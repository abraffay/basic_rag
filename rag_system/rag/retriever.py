import numpy as np

from rank_bm25 import BM25Okapi

from langchain_community.vectorstores import FAISS


class HybridRetriever:

    def __init__(
        self,
        chunks,
        embeddings
    ):

        if not chunks:
            raise ValueError(
                "Cannot build retriever: no document chunks. "
                "Check that PDFs in the docs folder contain extractable text."
            )

        self.chunks = chunks

        self.db = FAISS.from_documents(
            chunks,
            embeddings
        )

        self.vector_retriever = (
            self.db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 10,
                    "fetch_k": 30
                }
            )
        )

        tokenized = [
            doc.page_content.lower().split()
            for doc in chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query
    ):

        query_tokens = (
            query.lower().split()
        )

        bm25_scores = (
            self.bm25.get_scores(
                query_tokens
            )
        )

        top_idx = (
            np.argsort(bm25_scores)[::-1][:10]
        )

        bm25_docs = [
            self.chunks[i]
            for i in top_idx
        ]

        vector_docs = (
            self.vector_retriever.invoke(
                query
            )
        )

        merged = (
            bm25_docs
            + vector_docs
        )

        seen = set()

        unique_docs = []

        for doc in merged:

            text = doc.page_content

            if text not in seen:

                unique_docs.append(doc)

                seen.add(text)

        print(unique_docs)
        print("--------------------------------")
        return unique_docs