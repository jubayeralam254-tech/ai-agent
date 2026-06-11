import asyncio
from database import async_engine
from models import Base

async def reset():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print('Done')

asyncio.run(reset())