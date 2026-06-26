from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from schemas import (
    QueryRequest, QueryResponse,
    UserRegister, UserResponse,
    LoginRequest, Token,
    TicketResponse,
)
from database import init_db, get_db
from agent import run_support_agent
from models import User, Organization, Ticket
from auth import hash_password, verify_password, create_access_token, get_current_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing Database...")
    await init_db()
    yield
    print("Shutting down...")


app = FastAPI(
    title="AI Support Operations Platform",
    description="Enterprise-grade multi-tenant AI support routing API with JWT auth.",
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

@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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