"""Consistency tripwire for community-pool tagging (prompt rule 8 + ingest guard).

Reports Pool-classified photos whose tags carry amenity signals but lack the
"community pool" tag — the failure mode that let clubhouse pools count as
private pools. After the historical re-vision cleanup this should stay at ~0;
a rising count means the vision prompt drifted or the ingest guard regressed.
Read-only: prints offenders, changes nothing.

Run:  docker exec realestatesearch-app-1 python -m src.data.audit_pool_tags
"""

from __future__ import annotations

import asyncio
import logging

from src.data.database import close_pool, get_pool

logger = logging.getLogger(__name__)

# WIDE net on purpose — this is a report for a human, so unlike the ingest
# guard it includes the lower-precision signals too (a reviewer can dismiss
# a private tennis court; the ingest guard must not guess).
_AUDIT_SIGNALS = [
    "clubhouse", "fitness center", "amenity", "onsite", "lap lanes",
    "cabana", "tennis court", "pickleball", "resort-style",
]

# Unit properties (condo / unit-numbered townhome) should essentially never
# have an unverified non-community Pool photo — the ingest verification tags
# them at write time. Anything here survived every layer.
_UNIT_QUERY = """
SELECT p.guid, p.street, p.city, p.home_type, ri.features_text
FROM room_instances ri
JOIN properties p ON p.id = ri.property_id
WHERE ri.room_type = 'Pool'
  AND NOT EXISTS (SELECT 1 FROM unnest(ri.features) f WHERE f ILIKE '%community%pool%')
  AND NOT ('private pool' = ANY(ri.features))
  AND (p.home_type = 'CONDO'
       OR (p.home_type IN ('TOWNHOUSE', 'MULTI_FAMILY')
           AND p.street ~* '\\y(apt|unit)\\y|#'))
ORDER BY p.city, p.street
"""

_QUERY = """
SELECT p.guid, p.street, p.city, ri.features_text
FROM room_instances ri
JOIN properties p ON p.id = ri.property_id
WHERE ri.room_type = 'Pool'
  AND NOT EXISTS (SELECT 1 FROM unnest(ri.features) f WHERE f ILIKE '%community%pool%')
  AND NOT ('private pool' = ANY(ri.features))  -- verified private by re-vision
  AND EXISTS (SELECT 1 FROM unnest(ri.features) f
              WHERE f ILIKE ANY(SELECT '%' || s || '%' FROM unnest($1::text[]) s))
ORDER BY p.city, p.street
"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(_QUERY, _AUDIT_SIGNALS)
    async with pool.acquire() as conn:
        unit_rows = await conn.fetch(_UNIT_QUERY)
    if unit_rows:
        props = {r["guid"] for r in unit_rows}
        logger.info(
            f"WARNING: {len(unit_rows)} unverified Pool photo(s) on {len(props)} UNIT "
            f"properties (condo/unit-numbered) — ingest verification should have tagged these:"
        )
        for r in unit_rows[:20]:
            logger.info(f"  {r['guid'][:8]} {r['street']}, {r['city']} [{r['home_type']}]")
        if len(unit_rows) > 20:
            logger.info(f"  ... and {len(unit_rows) - 20} more")
    else:
        logger.info("OK: no unverified Pool photos on unit properties.")
    if not rows:
        logger.info("OK: no amenity-signal Pool photos without the community tag.")
    else:
        props = {r["guid"] for r in rows}
        logger.info(
            f"WARNING: {len(rows)} Pool photo(s) on {len(props)} properties carry "
            f"amenity signals but no 'community pool' tag:"
        )
        for r in rows[:30]:
            logger.info(f"  {r['guid'][:8]} {r['street']}, {r['city']}: {r['features_text'][:100]}")
        if len(rows) > 30:
            logger.info(f"  ... and {len(rows) - 30} more")
    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
