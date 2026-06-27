import asyncio
import os
from contextlib import asynccontextmanager

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import (
    QueryRequest, QueryResponse,
    UserRegister, UserResponse,
    LoginRequest, Token,
    TicketResponse,
    DocumentUploadRequest, DocumentResponse,
)
from database import init_db, get_db
from agent import run_support_agent
from models import User, Organization, Ticket, Document, DocumentChunk
from auth import hash_password, verify_password, create_access_token, get_current_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Database...")
    await init_db()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    yield
    print("Shutting down...")


app = FastAPI(
    title="AI Support Operations Platform",
    description="Enterprise-grade multi-tenant AI support API with JWT auth.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── ROLE GUARD ───────────────────────────────────────────────────────────────

def require_role(*roles: str):
    async def checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role: {list(roles)}"
            )
        return current_user
    return checker


# ─── HEALTH ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "error"
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "service": "AI Support API",
        "version": "2.0.0",
        "database": db_status,
    }


# ─── AUTH ─────────────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=UserResponse, status_code=201)
async def register(user_data: UserRegister, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == user_data.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    org = await db.get(Organization, user_data.organization_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    user = User(
        organization_id=user_data.organization_id,
        email=user_data.email,
        role=user_data.role.value,
        hashed_password=hash_password(user_data.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@app.post("/auth/login", response_model=Token)
async def login(login_data: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == login_data.email))
    user = result.scalar_one_or_none()

    if not user or not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({
        "sub": str(user.id),
        "org_id": str(user.organization_id),
        "role": user.role,
    })
    return Token(
        access_token=token,
        token_type="bearer",
        user_id=user.id,
        organization_id=user.organization_id,
        role=user.role,
    )


# ─── DOCUMENTS ────────────────────────────────────────────────────────────────

@app.post("/api/v1/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(
    request: DocumentUploadRequest,
    current_user: User = Depends(require_role("admin", "agent")),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a new document to the knowledge base.
    Generates a 3072-dim Gemini embedding and stores it in pgvector.
    Admin/Agent only.
    """
    # 1. Generate embedding
    try:
        loop = asyncio.get_running_loop()
        embed_result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model="models/gemini-embedding-001",
                content=request.content,
            )
        )
        embedding_vector = embed_result["embedding"]
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding generation failed: {str(e)}"
        )

    # 2. Save Document
    doc = Document(
        organization_id=current_user.organization_id,
        title=request.title,
        source_url=request.source_url,
    )
    db.add(doc)
    await db.flush()

    # 3. Save DocumentChunk with embedding
    embedding_str = "[" + ",".join(map(str, embedding_vector)) + "]"
    await db.execute(
        text("""
            INSERT INTO document_chunks
                (id, document_id, content, chunk_index, embedding, created_at, updated_at)
            VALUES
                (gen_random_uuid(), :doc_id, :content, 0, :embedding::vector, NOW(), NOW())
        """),
        {
            "doc_id": str(doc.id),
            "content": request.content,
            "embedding": embedding_str,
        }
    )

    await db.commit()
    await db.refresh(doc)
    return doc


@app.get("/api/v1/documents", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents in the organization's knowledge base."""
    result = await db.execute(
        select(Document)
        .where(Document.organization_id == current_user.organization_id)
        .order_by(Document.created_at.desc())
    )
    return result.scalars().all()


# ─── QUERY ────────────────────────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse)
async def ask_agent(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        agent_result = await run_support_agent(
            query=request.query,
            organization_id=current_user.organization_id,
        )

        ticket = Ticket(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            query=request.query,
            response=agent_result["response"],
            needs_human=agent_result["needs_human"],
        )
        db.add(ticket)
        await db.commit()
        await db.refresh(ticket)

        return QueryResponse(
            result=agent_result["response"],
            needs_human=agent_result["needs_human"],
            agent_steps=agent_result["steps"],
            ticket_id=ticket.id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent workflow failed: {str(e)}",
        )


# ─── TICKETS ──────────────────────────────────────────────────────────────────

@app.get("/api/v1/tickets", response_model=list[TicketResponse])
async def get_tickets(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Ticket)
        .where(Ticket.organization_id == current_user.organization_id)
        .order_by(Ticket.created_at.desc())
        .limit(50)
    )
    return result.scalars().all()