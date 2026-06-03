from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.llms import Ollama

loader = PyMuPDFLoader("data.pdf")

documents = loader.load()

print("Pages:", len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever()


llm = Ollama(model="llama3")



prompt = ChatPromptTemplate.from_template("""
Use ONLY the provided context.

Context:
{context}

Question:
{question}

Answer clearly.
""")


def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


while True:
    query = input("\nAsk PDF: ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)

    print("\nAnswer:\n")
    print(response)