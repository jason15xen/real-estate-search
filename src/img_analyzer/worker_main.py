"""
Standalone worker process.

Runs the background worker on its own — fully separated from the FastAPI web
tier. Both processes connect to the same Postgres; coordination happens via
the raw_properties queue + Postgres NOTIFY (`feature_change` channel).

Started by the `worker` service in docker-compose.yml:
    python -m src.img_analyzer.worker_main
"""
from __future__ import annotations

import asyncio
import logging
import signal

from config.settings import settings
from src.data.database import close_pool, get_pool
from src.img_analyzer.worker import run_worker_forever

logging.basicConfig(
    level=settings.log_level,
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

    await close_pool()
    logger.info("Worker process stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
