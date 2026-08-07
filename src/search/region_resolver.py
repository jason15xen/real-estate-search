"""Resolve the place a query searched INSIDE to a row of the regions table.

Geo-location contract: /search returns regionId + canonical regionName + the
boundary ONLY when that region's polygon actually filtered the results
(resolve_search_region succeeded and the region has geometry). Every fallback —
no polygon data, subdivision/neighborhood searches, ambiguous names — returns
regionId null, with the parsed place name as a bare label. regionId is a promise
that the pins are that region's geometry; never a guess.
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
    # '3' at the end: users say bare "brevard" for the county, which the parser
    # files under city; the only type-0 'Brevard' is a town in NC, so without the
    # county fall-through the obvious local meaning would never resolve.
    "locality": ("0", "1", "3"),
    "city": ("0", "1", "3"),
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
    """The PARSED area name — the display label when no region polygon filtered the
    search (the canonical name then comes from resolve_search_region instead)."""
    target = search_area_target(criteria)
    if target is None:
        return None
    field, value, _ = target
    # State expanded to its full name — "Florida" geocodes where "FL" is ambiguous.
    return expand_state(value) if field == "state" else value


async def resolve_search_region(pool: asyncpg.Pool, criteria) -> dict | None:
    """PRE-FILTER region resolution for geo-location search: map the searched place
    name to one regions row BEFORE any properties are matched, so its polygon can
    replace name matching entirely. Returns
    {"region_id", "region_name", "has_geom"} or None.

    No matched results exist yet, so candidates narrow by the query's own
    state/parent-city context, then by which candidate's POLYGON actually contains
    catalog properties ("Rockledge" with no state: the FL polygon holds 300+ homes,
    the GA one zero). Ambiguous, unknown, or polygon-less -> None, and the caller
    falls back to name matching with regionId null. Never raises."""
    try:
        target = search_area_target(criteria)
        if target is None:
            return None
        field, value, crit = target
        state_code = abbrev_state(crit.state).upper() if crit.state and crit.state.strip() else None
        parent_city = crit.city if field in ("neighborhood", "district", "locality") else None

        async with pool.acquire() as conn:
            for region_type in _FIELD_REGION_TYPES[field]:
                names = [value]
                if field == "state":
                    names = [expand_state(value)]
                elif region_type == "3" and not value.lower().endswith("county"):
                    names.append(f"{value} County")  # bare "Brevard" -> the county row

                candidates = await conn.fetch(
                    """
                    SELECT regionid, regionname, statecode,
                           (geom IS NOT NULL) AS has_geom
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
                if len(candidates) == 1:
                    chosen = candidates[0]
                else:
                    # Tie-break by polygon population: which candidate's boundary
                    # actually contains catalog homes.
                    populated = []
                    for c in (c for c in candidates if c["has_geom"]):
                        n = await conn.fetchval(
                            "SELECT count(*) FROM properties p, regions r "
                            "WHERE r.regionid = $1 AND ST_Covers(r.geom, p.geom)",
                            c["regionid"],
                        )
                        if n:
                            populated.append((n, c))
                    if len(populated) > 1:
                        return None  # genuinely ambiguous -> no polygon mode
                    if not populated:
                        continue  # nothing usable at this level, try the next type
                    chosen = populated[0][1]
                if not chosen["has_geom"]:
                    # A polygon-less match can't power geo filtering; try the next
                    # type first (bare "brevard" hits the polygon-less NC town at
                    # type '0' but Brevard County's polygon at '3') before falling
                    # back to name matching.
                    continue
                return {
                    "region_id": chosen["regionid"],
                    "region_name": f"{chosen['regionname']}, {chosen['statecode']}",
                    "has_geom": True,
                }
        return None
    except Exception:
        logger.exception("Pre-filter region resolution failed")
        return None


async def keep_majority_location(
    pool: asyncpg.Pool, criteria, property_ids: list[int]
) -> list[int]:
    """When a place NAME was searched WITHOUT a state, same-named places in different
    states can all match ("Rockledge" hits FL and GA listings) — one map, two places.
    Trim the results to the single state holding the most matches so the response
    describes one location; ties break deterministically (highest count, then
    statecode A→Z) so identical requests return identical results.

    Untouched cases: no place searched (feature-only queries legitimately span the
    catalog), a state was given (the hard filter already pinned it), or the results
    already sit in one state. Never raises — trimming is enrichment, so failures
    return the ids unchanged."""
    target = search_area_target(criteria)
    if target is None or not property_ids:
        return property_ids
    _, _, crit = target
    if crit.state and crit.state.strip():
        return property_ids

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, state FROM properties WHERE id = ANY($1)", property_ids
            )
        by_state: dict[str, list[int]] = {}
        for r in rows:
            by_state.setdefault(abbrev_state(r["state"]).upper(), []).append(r["id"])
        if len(by_state) <= 1:
            return property_ids
        winner = min(by_state, key=lambda s: (-len(by_state[s]), s))
        logger.info(
            f"Majority-location trim: keeping {winner} "
            f"({len(by_state[winner])}/{len(property_ids)} results; "
            f"dropped states: {sorted(s for s in by_state if s != winner)})"
        )
        return by_state[winner]
    except Exception:
        logger.exception("Majority-location trim failed — keeping all results")
        return property_ids


