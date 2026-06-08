import os
from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from models import Base

# Load the database URL from the environment.
# Ensure you are using the async driver: postgresql+asyncpg://...
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:jubojubo67@localhost:5432/ai_support_db"
)

# 1. Create the async engine
# pool_pre_ping checks the connection validity before returning it from the pool
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# 2. Configure the async session factory
# expire_on_commit=False is strictly required for AsyncSession to prevent lazy-loading issues after commit
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# 3. FastAPI Dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides an async database session per request.
    Automatically handles closing the session after the request finishes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# 4. Startup Event / Provisioning
async def init_db() -> None:
    """
    Initialize the database. 
    Should be called during application startup (FastAPI lifespan event).
    """
    async with async_engine.begin() as conn:
        # run_sync creates tables within the async context
        await conn.run_sync(Base.metadata.create_all)
