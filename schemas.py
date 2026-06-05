from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str