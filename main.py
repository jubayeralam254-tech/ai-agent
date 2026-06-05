from fastapi import FastAPI
from schemas import QueryRequest, QueryResponse
from agent import run_agent

app = FastAPI(title="AI Agent API")

@app.get("/")
def root():
    return {"message": "AI Agent is running"}

@app.post("/agent", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    result = run_agent(request.query)
    return QueryResponse(result=result)