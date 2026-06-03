import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="Multi PDF RAG",
)

st.title(
    "📚 Multi PDF Chat"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


embeddings = HuggingFaceEmbeddings(
    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"
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
Previous conversation:

{history}

Context:

{context}

Question:

{question}

Answer clearly.
""")


def format_docs(docs):

    return "\n\n".join(
        d.page_content
        for d in docs
    )


def format_history():

    rows = []

    for msg in st.session_state.messages:

        rows.append(
            f"{msg['role']}: {msg['content']}"
        )

    return "\n".join(rows)


for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):

        st.write(
            msg["content"]
        )


query = st.chat_input(
    "Ask your documents..."
)


if query:

    st.session_state.messages.append(
        {
            "role":
            "user",

            "content":
            query
        }
    )

    docs = retriever.invoke(
        query
    )

    context = format_docs(
        docs
    )

    history = format_history()

    final_prompt = prompt.invoke(
        {
            "history":
            history,

            "context":
            context,

            "question":
            query
        }
    )

    answer = llm.invoke(
        final_prompt
    )

    st.session_state.messages.append(
        {
            "role":
            "assistant",

            "content":
            answer
        }
    )

    st.rerun()


