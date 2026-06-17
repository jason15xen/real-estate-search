"""Targeted re-vision backfill for properties.pool_covered.

Re-analyzes ONLY the images where a pool is visible (Pool-room images + Exterior
images that mention a pool) with a focused covered/uncovered/community prompt,
then aggregates one verdict per property. This corrects pool_covered accurately
without reprocessing the whole image set (~1.1k of 13.6k images).

Run:  python -m src.img_analyzer.reclassify_pool_coverage
      python -m src.img_analyzer.reclassify_pool_coverage --pool-only   # skip Exterior
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from config.settings import settings
from src.data.database import get_pool
from src.llm_client import get_async_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_CONCURRENCY = 12

_FOCUSED_PROMPT = """\
You are shown ONE real-estate photo that may contain a swimming pool. Decide the
pool situation for THIS listing. Choose exactly one:
- "covered"   — the pool WATER is under a screen enclosure / pool cage / roof /
                lanai roof / solid or retractable cover.
- "uncovered" — a private pool that is open to the sky. A fence or railing around
                the pool is NOT a cover; a covered patio/deck/lanai/cabana NEXT TO
                the pool does NOT make the pool covered — judge the water itself.
- "community" — the pool is clearly a shared / resort / community amenity (large
                resort-style pool, rows of lounge chairs, cabanas, multiple
                umbrellas), not the home's own private pool.
- "none"      — no swimming pool is actually visible in this image.

Return STRICT JSON: {"coverage": "covered|uncovered|community|none"}. No prose."""

async def _classify_image(client, sem, url: str) -> str:
    """Focused vision verdict for one image: covered|uncovered|community|none.
    Any failure (dead URL, parse error) → 'none' (no usable signal)."""
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=settings.openai_model,
                max_completion_tokens=200,
                messages=[
                    {"role": "system", "content": _FOCUSED_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                    ]},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))
            v = json.loads(raw).get("coverage", "none")
            return v if v in ("covered", "uncovered", "community", "none") else "none"
        except Exception as e:  # noqa: BLE001 — best-effort; one bad image must not abort the run
            logger.warning("classify failed for %s: %s", url, e)
            return "none"


def _aggregate(verdicts: list[str], tags: set[str]) -> bool:
    """One boolean per property: TRUE iff any image shows the pool itself covered
    (screen/cage/roof). Uncovered/community/none all map to FALSE — "uncovered" is
    derived at search time as (has a pool) AND NOT has_covered_pool, so it isn't
    stored here. `tags` is unused now but kept for signature stability."""
    return "covered" in verdicts


async def main(pool_only: bool) -> None:
    pool = await get_pool()
    client = get_async_client()
    sem = asyncio.Semaphore(_CONCURRENCY)

    room_filter = "ri.room_type = 'Pool'" if pool_only else (
        "(ri.room_type = 'Pool' OR (ri.room_type = 'Exterior' "
        "AND EXISTS (SELECT 1 FROM unnest(ri.features) f WHERE f ILIKE '%pool%')))"
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT p.id, p.guid, ri.photo_url, ri.features
            FROM properties p JOIN room_instances ri ON ri.property_id = p.id
            WHERE ri.photo_url IS NOT NULL AND {room_filter}
        """)

    # Group images + collect each property's pool-ish tags.
    by_prop: dict[int, dict] = {}
    for r in rows:
        d = by_prop.setdefault(r["id"], {"guid": r["guid"], "urls": [], "tags": set()})
        d["urls"].append(r["photo_url"])
        d["tags"].update(f.lower() for f in (r["features"] or []))
    logger.info("Re-visioning %d images across %d properties (pool_only=%s)",
                len(rows), len(by_prop), pool_only)

    # Classify every image once, concurrently.
    urls = [u for d in by_prop.values() for u in d["urls"]]
    results = await asyncio.gather(*[_classify_image(client, sem, u) for u in urls])
    url_verdict = dict(zip(urls, results))

    # Aggregate per property and write back.
    updates = {}
    for pid, d in by_prop.items():
        verdicts = [url_verdict[u] for u in d["urls"]]
        updates[pid] = _aggregate(verdicts, d["tags"])

    async with pool.acquire() as conn:
        for pid, verdict in updates.items():
            await conn.execute("UPDATE properties SET has_covered_pool = $2 WHERE id = $1", pid, verdict)

    covered = sum(1 for v in updates.values() if v)
    logger.info("Done. has_covered_pool within re-visioned set: %d covered / %d not",
                covered, len(updates) - covered)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-only", action="store_true",
                    help="Only re-vision Pool-room images (skip Exterior pool images).")
    args = ap.parse_args()
    asyncio.run(main(args.pool_only))
