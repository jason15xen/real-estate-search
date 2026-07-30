"""Resolve the place a query searched INSIDE to a row of the regions table.

The /search response has always returned the area as a bare name ("Downtown"), but
names are not unique — "Downtown" appears 137 times in the catalog — so the client
also needs the region's unique id. Resolution matches the name against regions rows
of the types implied by WHICH LocationCriterion field the name came from (see
_FIELD_REGION_TYPES), narrows duplicates with the criterion's own state/city context,
and as a last resort with the states/cities of the properties the search actually
matched. Still ambiguous -> no id (never a guess).
"""

import logging

import asyncpg

from src.data.us_states import abbrev_state, expand_state
from src.models.search import LocationCriterion

logger = logging.getLogger(__name__)

# regions.regiontype values a given LocationCriterion field can map to, tried in
# order. city/locality fall through '0' -> '1' because the parser deliberately
# files ANY doubtful sub-city place under city (its prompt: "when in doubt, put
# the place here") — so "Silver Lake" arrives as city even when the real region
# row is a neighborhood. Mirrors the hard filter, which matches city across all
# place columns for the same reason.
_FIELD_REGION_TYPES: dict[str, tuple[str, ...]] = {
    "neighborhood": ("1",),
    "district": ("1",),
    "locality": ("0", "1"),
    "city": ("0", "1"),
    "county": ("3",),
    "state": ("4",),
}


def search_area_target(criteria) -> tuple[str, str, LocationCriterion] | None:
    """(field, value, criterion) for the place the user is searching INSIDE, or None
    when the query names no such place OR is RELATIONAL — "near X", "between X and Y"
    — because those cover a ring/corridor, not X's own polygon. Most-specific
    location field wins."""
    for c in criteria:
        if isinstance(c, LocationCriterion):
            for field in ("neighborhood", "locality", "district", "city", "county"):
                value = getattr(c, field)
                if value and str(value).strip():
                    return field, str(value).strip(), c
            if c.state and c.state.strip():
                return "state", c.state.strip(), c
    return None


def extract_search_area(criteria) -> str | None:
    """The PARSED area name — the fallback display name when no regions row resolves
    (the canonical name then comes from resolve_region instead)."""
    target = search_area_target(criteria)
    if target is None:
        return None
    field, value, _ = target
    # State expanded to its full name — "Florida" geocodes where "FL" is ambiguous.
    return expand_state(value) if field == "state" else value


async def resolve_region(
    pool: asyncpg.Pool, criteria, property_ids: list[int]
) -> dict | None:
    """{"region_id": regions.regionid, "region_name": "<RegionName>, <StateCode>"}
    for the searched area (e.g. {"region_id": 811179, "region_name": "Aquarian
    Acres, KS"}), or None (no area, unknown name, or ambiguous after narrowing).
    property_ids are the search's matched properties: their states/cities break
    ties between same-named candidates and veto candidates that contradict every
    result. Never raises: region info is enrichment, so any failure (e.g. regions
    table not yet imported) degrades to None instead of failing the whole /search."""
    try:
        row = await _resolve_region(pool, criteria, property_ids)
    except Exception:
        logger.exception("Region resolution failed — returning no region")
        return None
    if row is None:
        return None
    return {
        "region_id": row["regionid"],
        # Canonical catalog spelling, not the user's — "rockledge" -> "Rockledge, FL".
        "region_name": f"{row['regionname']}, {row['statecode']}",
    }


async def _resolve_region(
    pool: asyncpg.Pool, criteria, property_ids: list[int]
) -> asyncpg.Record | None:
    target = search_area_target(criteria)
    if target is None:
        return None
    field, value, crit = target

    # upper(): regions.statecode is 'FL'; the LLM may emit 'fl' and abbrev_state
    # passes 2-letter values through unchanged.
    state_code = abbrev_state(crit.state).upper() if crit.state and crit.state.strip() else None
    # Parent city narrows neighborhoods ("Downtown" -> the one in the queried city).
    # Only valid against type-'1' rows: for other types regions.city is NOT a city
    # (it's the state code, or '' for states), so it would falsely exclude them.
    parent_city = crit.city if field in ("neighborhood", "district", "locality") else None

    async with pool.acquire() as conn:
        # Where the matched properties actually are — used both to break ties between
        # same-named candidates and to reject a lone candidate that contradicts every
        # result (a single match is not proof it's the RIGHT place: "Rockledge" filed
        # as a neighborhood matches only the one in Jenkintown PA, while all the
        # result pins are in FL). Empty result set = no evidence either way.
        prop_states: set[str] = set()
        prop_cities: set[str] = set()
        if property_ids:
            rows = await conn.fetch(
                "SELECT DISTINCT state, city FROM properties WHERE id = ANY($1)",
                property_ids,
            )
            prop_states = {abbrev_state(r["state"]).upper() for r in rows}
            prop_cities = {r["city"].strip().lower() for r in rows if r["city"]}

        for region_type in _FIELD_REGION_TYPES[field]:
            names = [value]
            if field == "state":
                # Type-4 rows: regionname is the full state name, statecode the code.
                names = [expand_state(value)]
            elif field == "county" and not value.lower().endswith("county"):
                names.append(f"{value} County")  # criterion may carry "Brevard" bare

            candidates = await conn.fetch(
                """
                SELECT regionid, regionname, statecode, city
                FROM regions
                WHERE regiontype = $1
                  AND lower(regionname) = ANY($2)
                  AND ($3::text IS NULL OR statecode = $3)
                  AND ($4::text IS NULL OR lower(city) = lower($4))
                """,
                region_type,
                [n.lower() for n in names],
                state_code,
                parent_city if region_type == "1" else None,
            )
            if not candidates:
                continue

            narrowed = (
                [c for c in candidates if c["statecode"].upper() in prop_states]
                if prop_states else list(candidates)
            )
            if len(narrowed) > 1 and prop_cities:
                # Duplicates within one state (e.g. two 'Downtown' neighborhoods):
                # keep those whose parent city matches a result's city, if any do.
                by_city = [c for c in narrowed if c["city"].strip().lower() in prop_cities]
                narrowed = by_city or narrowed
            if len(narrowed) == 1:
                return narrowed[0]
            if not narrowed:
                # Every candidate at this level is in a state none of the results
                # are in — same-named places elsewhere ("Silver Lake" towns in
                # IN/KS/MN/OH while the pins are in FL). Fall through to the next
                # region type rather than mislabel or give up early.
                continue
            logger.info(
                f"Region '{value}' ({field}) ambiguous: {len(candidates)} candidates, "
                f"{len(narrowed)} after property narrowing — returning no regionid"
            )
            return None
    return None
