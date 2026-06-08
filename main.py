from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from schemas import QueryRequest, QueryResponse
from database import init_db
from agent import run_support_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan manager. 
    Startup: Initialize database connection and schemas/pgvector.
    Shutdown: Clean up resources.
    """
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

# Configure CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific origins in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Simple health check endpoint for Render/Kubernetes."""
    return {"status": "healthy", "service": "AI Support API"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def ask_agent(
    request: QueryRequest
):
    """
    Main entry point for customer queries.
    Expects a query and an organization_id for tenant-isolated RAG.
    """
    try:
        # We pass the query and Org ID down into our LangGraph async entry point
        agent_result = await run_support_agent(
            query=request.query, 
            organization_id=request.organization_id
        )
        
        return QueryResponse(
            result=agent_result["response"],
            agent_steps=agent_result["steps"]
        )
    except Exception as e:
        # In a real SaaS, we would log this structured exception to Sentry/Datadog
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow failed: {str(e)}"
        )