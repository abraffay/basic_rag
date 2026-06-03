from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "AI is used in healthcare for diagnosis.",
    "Machine learning helps systems learn from data.",
    "Deep learning is a subset of machine learning.",
    "Cricket is very popular in Pakistan."
]
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever()

llm = OllamaLLM(model="llama3")

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n".join([d.page_content for d in docs])

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


while True:
    query = input("\nAsk: ")

    response = rag_chain.invoke(query)

    print("\nAnswer:\n", response)