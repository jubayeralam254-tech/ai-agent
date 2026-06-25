from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from schemas import QueryRequest, QueryResponse
from database import init_db
from agent import run_support_agent

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Database Connections and PGVector...")
    await init_db()
    yield
    print("Shutting down AI Support Operations Platform...")

app = FastAPI(
    title="AI Support Operations Platform",
    description="Enterprise-grade AI support routing and resolution API.",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Support API"}

@app.post("/api/v1/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def ask_agent(request: Request, body: QueryRequest):
    try:
        agent_result = await run_support_agent(
            query=body.query,
            organization_id=body.organization_id
        )
        return QueryResponse(
            result=agent_result["response"],
            needs_human=agent_result["needs_human"],
            agent_steps=agent_result["steps"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow failed: {str(e)}"
        )
