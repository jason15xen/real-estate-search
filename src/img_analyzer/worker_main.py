"""Standalone worker process (separate from the FastAPI web tier); coordinates via the raw_properties queue and Postgres NOTIFY (feature_change)."""
from __future__ import annotations

import asyncio
import logging
import signal

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.img_analyzer.worker import run_worker_forever

logging.basicConfig(
    level=settings.log_level.upper(),  # tolerate LOG_LEVEL=info in .env
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    await get_pool()
    logger.info("Worker process: database pool initialized")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(signame: str) -> None:
        logger.info(f"Received {signame}, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig.name)

    worker_task = asyncio.create_task(run_worker_forever())
    await stop_event.wait()

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # Cancel the background POI refresh too — it can hold a pool connection through a
    # ~200s blocking Overpass fetch, and Pool.close() waits on every holder (SIGKILL
    # would hit before "stopped cleanly" otherwise).
    from src.img_analyzer import worker as worker_mod
    poi_task = worker_mod._poi_refresh_task
    if poi_task is not None and not poi_task.done():
        poi_task.cancel()
        try:
            await poi_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — shutdown best-effort
            pass

    await close_pool()
    logger.info("Worker process stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
