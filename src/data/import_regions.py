"""Load the region catalog (src/data/region/regions_data.sql) into the regions table.

The data file is 72k `INSERT INTO Regions ...` statements exported from the source
system. Executing them one-by-one is slow, so this parses the rows out and bulk-loads
them with COPY. The load is idempotent: it replaces the whole table inside one
transaction (readers never see it empty).

Run:  python -m src.data.import_regions

The table itself is defined in schema/postgresql.sql; it is also created here so an
existing database (initdb already ran) picks it up without a manual migration.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from src.data.database import get_pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_FILE = Path(__file__).parent / "region" / "regions_data.sql"

# One INSERT per line; string values use SQL escaping ('' for a literal quote).
_ROW_RE = re.compile(
    r"VALUES \('([0-9A-Fa-f-]{36})', (\d+), '(\d)', '((?:[^']|'')*)', '([^']*)', '((?:[^']|'')*)'\);"
)

_DDL = """
CREATE TABLE IF NOT EXISTS regions (
    id          UUID PRIMARY KEY,
    regionid    INTEGER NOT NULL UNIQUE,
    regiontype  TEXT NOT NULL,
    regionname  TEXT NOT NULL,
    statecode   TEXT NOT NULL,
    city        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_regions_type_name ON regions (regiontype, lower(regionname));
CREATE INDEX IF NOT EXISTS idx_regions_statecode ON regions (statecode);
"""


def _unescape(value: str) -> str:
    return value.replace("''", "'")


def parse_rows(text: str) -> list[tuple]:
    """(id, regionid, regiontype, regionname, statecode, city) per data line."""
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        m = _ROW_RE.search(line)
        if not m:
            # Refuse to half-load: a malformed line means the regex (or file format)
            # is wrong, and silently skipping would drop regions nobody would miss
            # until a search fails to resolve.
            raise ValueError(f"Unparseable line {line_no}: {line[:120]}")
        guid, region_id, rtype, name, state, city = m.groups()
        rows.append((guid, int(region_id), rtype, _unescape(name), state, _unescape(city)))
    return rows


async def import_regions() -> int:
    text = DATA_FILE.read_text(encoding="utf-8-sig")  # -sig: file starts with a BOM
    rows = parse_rows(text)
    logger.info(f"Parsed {len(rows)} regions from {DATA_FILE.name}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(_DDL)
        async with conn.transaction():
            await conn.execute("TRUNCATE regions")
            await conn.copy_records_to_table(
                "regions",
                records=rows,
                columns=["id", "regionid", "regiontype", "regionname", "statecode", "city"],
            )
        count = await conn.fetchval("SELECT count(*) FROM regions")
    logger.info(f"regions table now holds {count} rows")
    return count


if __name__ == "__main__":
    asyncio.run(import_regions())
