from fastapi import FastAPI
from pydantic import BaseModel
from rag.pipeline import answer_question

app = FastAPI()


class Question(BaseModel):
    query: str


@app.get("/")
def home():
    return {"status": "running"}

@app.post("/chat")
def chat(question: Question):
    answer = answer_question(question.query)
    return {
        "query": question.query,
        "answer": answer
    }