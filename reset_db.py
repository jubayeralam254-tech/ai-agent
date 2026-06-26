import asyncio
from sqlalchemy import text
from database import async_engine
from models import Base


async def reset():
    async with async_engine.begin() as conn:
        # Safe migration — existing data নষ্ট হবে না
        await conn.execute(text(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS hashed_password VARCHAR(255);"
        ))
        # tickets table নতুন তৈরি হবে
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Schema updated:")
    print("   - hashed_password column added to users")
    print("   - tickets table created")


asyncio.run(reset())