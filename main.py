# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
from rag import get_answer

app = FastAPI()

class Query(BaseModel):
    question: str

@app.get('/')
def hello():
    return {"message": "This is Faisal's PingCRM"}

@app.post("/ask")
async def ask(query: Query):
    # return {"answer": get_answer(query.question)}

    start_time = time.time()  # Start timer
    answer = get_answer(query.question)
    end_time = time.time()    # End timer

    duration = round(end_time - start_time, 2)
    print(f"⏱️ get_answer completed in {duration} seconds")

    return {
        "answer": answer,
        "time_taken_seconds": duration
    }
