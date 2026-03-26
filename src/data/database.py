"""
Database connection pool — shared across the application.
"""

import asyncio
import logging

import asyncpg

from config.settings import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        dsn = (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        # Retry connection up to 15 times (DB may still be initializing)
        for attempt in range(1, 16):
            try:
                _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)
                logger.info("Database pool connected")
                return _pool
            except (OSError, asyncpg.exceptions.ConnectionDoesNotExistError) as e:
                logger.warning(f"DB connection attempt {attempt}/15 failed: {e}")
                if attempt == 15:
                    raise
                await asyncio.sleep(2)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
