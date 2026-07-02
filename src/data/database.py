"""Shared database connection pool."""

import asyncio
import logging

import asyncpg

from config.settings import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        # Double-check lock.
        if _pool is not None:
            return _pool
        dsn = (
            f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
            f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        # Retry up to 15 times (DB may still be initializing).
        for attempt in range(1, 16):
            try:
                # Web tier reserves one conn for the LISTEN feature_change callback.
                _pool = await asyncpg.create_pool(dsn=dsn, min_size=5, max_size=20)
                logger.info("Database pool connected")
                return _pool
            except (
                OSError,
                asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.CannotConnectNowError,  # "the database system is starting up"
            ) as e:
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
