from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.llms import Ollama


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(
    search_kwargs={"k": 4}
)


llm = Ollama(
    model="llama3"
)


prompt = ChatPromptTemplate.from_template("""
Answer using context but only if the question is related to the context. you can look up for references to get more detailed response.

Context:
{context}

Question:
{question}

Answer:
""")


def format_docs(docs):

    output = []

    for d in docs:

        source = d.metadata.get(
            "source_file",
            "unknown"
        )

        text = (
            f"[FILE: {source}]\n"
            + d.page_content
        )

        output.append(text)

    return "\n\n".join(output)


rag_chain = (
    {
        "context":
            retriever
            | format_docs,

        "question":
            RunnablePassthrough()
    }

    | prompt

    | llm
)


while True:

    q = input(
        "\nAsk: "
    )

    if q == "exit":
        break

    answer = rag_chain.invoke(
        q
    )

    print("\n")
    print(answer)