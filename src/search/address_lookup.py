"""Exact-address search: a query that names a specific house — house number +
street (unit / city / state / ZIP optional) — scopes the search to that home.

Detection is deterministic (no LLM): a house number followed by a street name
that ends in a recognised street suffix. Anything else — a bare street name, a
neighbourhood, a ZIP — is an INCOMPLETE address and goes through the normal
search, so a lucky single result can never masquerade as an exact match.

The remainder of the query ("... with a pool") is parsed as usual and applied
STRICTLY to the address's homes (no auto-relaxation): "1845 Hidden Lake Dr with a
pool" on a home without a pool is an empty list, not that home. The response
flag `exactAddress` is true only when the query was a complete address AND
exactly one property survives every criterion — the signal for the UI to open
that home directly instead of showing a list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import asyncpg

# Canonical short forms; both the query and the stored street go through this.
_SYNONYMS = {
    "drive": "dr", "street": "st", "avenue": "ave", "road": "rd", "lane": "ln",
    "court": "ct", "circle": "cir", "boulevard": "blvd", "place": "pl",
    "terrace": "ter", "trail": "trl", "parkway": "pkwy", "highway": "hwy",
    "point": "pt", "cove": "cv", "causeway": "cswy", "square": "sq",
    "crossing": "xing", "bend": "bnd", "manor": "mnr", "vista": "vis",
    "plaza": "plz", "alley": "aly", "trace": "trce", "glen": "gln",
    "heights": "hts", "landing": "lndg", "ridge": "rdg", "shore": "shr",
    "view": "vw", "valley": "vly", "north": "n", "south": "s", "east": "e",
    "west": "w", "northeast": "ne", "northwest": "nw", "southeast": "se",
    "southwest": "sw", "apartment": "apt", "suite": "ste", "number": "unit",
    "estates": "ests", "island": "is",
}
_DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw"}
_UNIT_MARKERS = {"apt", "unit", "ste", "lot", "bldg"}
# Words that never belong to a street name — reject "2 story homes on Main St"
# (house number 2?) while accepting "2 Main St".
_NOT_STREET_WORDS = {
    "bedroom", "bedrooms", "bed", "beds", "bath", "baths", "bathroom", "bathrooms",
    "home", "homes", "house", "houses", "condo", "condos", "with", "without",
    "in", "near", "under", "over", "pool", "story", "stories", "sqft",
    "k", "and", "or", "for", "on", "at", "about",
    "min", "mins", "minute", "minutes", "hour", "hours", "mile", "miles",
    "car", "cars", "block", "blocks", "step", "steps", "year", "years",
}


def _tokens(text: str) -> list[str]:
    text = text.lower().replace("#", " unit ")
    text = re.sub(r"[^\w\s]", " ", text)          # hyphens separate too ("Sea-Breeze" == "Sea Breeze")
    return [_SYNONYMS.get(t, t) for t in text.split() if t]


# Zillow-style URL slug: "4875-Winchester-Dr-Titusville-FL-32780" — one hyphenated
# word that is a whole address. Expanded to spaces before scanning.
_SLUG_RE = re.compile(r"^\d{1,6}[A-Za-z]?(?:-[A-Za-z0-9']+){2,}$")
_HAS_LETTERS = re.compile(r"-[0-9]*[A-Za-z]")            # excludes "321-555-1234"


@dataclass
class AddressQuery:
    number: str
    street: list[str]        # normalized tokens after the number, up to the unit
    unit: str | None
    city: str | None
    state: str | None
    zip: str | None
    remainder: str           # the query with the address removed (may be "")
    matched_text: str


_SUFFIX_SET = {
    "dr", "st", "ave", "rd", "ln", "ct", "cir", "blvd", "way", "pl", "ter",
    "trl", "pkwy", "hwy", "loop", "run", "path", "pt", "cv", "isle", "key",
    "cswy", "sq", "row", "walk", "xing", "bnd", "mnr", "vis", "plz", "aly",
    "trce", "gln", "hts", "lndg", "rdg", "shr", "spur", "vw", "vly", "bay", "ests", "is",
}
_ROUTE_DESIGNATOR = re.compile(r"^(?:a1a|us-?1|\d{1,3})$")
_HOUSE_NUMBER = re.compile(r"^\d{1,6}[A-Za-z]?$")
# What may follow the street: unit, ", city", state, ZIP — anchored at the tail.
_TAIL_RE = re.compile(
    r"^(?:\s*,?\s*(?:apt|unit|suite|ste|lot|#)\s*\.?\s*(?P<unit>[a-z0-9-]+))?"
    # ", City" — or, without a comma, "City" only when a state follows it
    # ("... Dr Titusville FL 32780", the expanded slug form).
    r"(?:\s*,\s*(?P<city>[a-z][a-z .'-]*?)(?=\s*(?:,|$)|\s+(?:fl|florida)\b|\s+\d{5}\b)"
    r"|\s+(?P<city2>[a-z][a-z .'-]*?)(?=\s+(?:fl|florida)\b|\s+\d{5}\b|$))?"
    r"(?:\s*,?\s*(?P<state>fl|florida)\b)?"
    r"(?:\s+(?P<zip>\d{5})\b)?",
    re.IGNORECASE,
)


# Known city names (normalized token tuples), loaded from the catalog and cached:
# "Merritt Island" / "Palm Bay" contain street-suffix words, so without this the
# comma-less form "… Dr Merritt Island FL" would absorb the city into the street.
_CITY_CACHE: dict = {"at": 0.0, "cities": frozenset()}
_CITY_TTL_SEC = 600


async def known_cities(pool: asyncpg.Pool) -> frozenset[tuple[str, ...]]:
    import time
    if time.monotonic() - _CITY_CACHE["at"] < _CITY_TTL_SEC and _CITY_CACHE["cities"]:
        return _CITY_CACHE["cities"]
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT DISTINCT city FROM properties WHERE city IS NOT NULL")
        cities = frozenset(tuple(_tokens(r["city"])) for r in rows if _tokens(r["city"]))
        _CITY_CACHE.update(at=time.monotonic(), cities=cities)
    except Exception:  # noqa: BLE001 — detection still works, just without city awareness
        pass
    return _CITY_CACHE["cities"]


def _city_ahead(words, j: int, cities: frozenset[tuple[str, ...]]) -> bool:
    """True if the words starting at index j spell a known city (1–3 words)."""
    if not cities:
        return False
    acc: list[str] = []
    for k in range(j, min(j + 3, len(words))):
        acc.extend(_tokens(words[k][0]))
        if tuple(acc) in cities:
            return True
        if len(acc) >= 3:
            break
    return False


def parse_address(query: str, cities: frozenset[tuple[str, ...]] = frozenset()) -> AddressQuery | None:
    """Return the complete address found in `query`, or None (incomplete / no address).

    Token scan rather than one regex: from each house-number-shaped word, walk
    forward collecting street words and remember the LONGEST span that ends in a
    street suffix. Street names often contain suffix-like words themselves
    ("Landing Dr", "Sykes Point Ln", "Tree Ridge Ln NE" — 188 catalog homes), so
    the first suffix seen is usually not the end of the street. A stop word, a
    unit marker or a stray number ends the walk; a lone suffix ("3 bay garage",
    "4 point inspection") is never a street."""
    query = " ".join(w.replace("-", " ") if _SLUG_RE.match(w) and _HAS_LETTERS.search(w) else w
                     for w in (query or "").split())
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", query)]
    for i, (raw, s, _) in enumerate(words):
        if not _HOUSE_NUMBER.match(raw.strip(",.;")):
            continue
        if s > 0 and query[s - 1] in "$-":          # "$300,000", "555-1234"
            continue
        name: list[str] = []
        best: tuple[int, int] | None = None          # (word index, token count)
        stop = False
        for j in range(i + 1, min(i + 8, len(words))):
            w = words[j][0]
            toks = _tokens(w)                        # "J.A." -> ["j","a"], "#413" -> ["unit","413"]
            if not toks or toks[0] in _UNIT_MARKERS:
                break                                # unit part begins
            if best is not None and _city_ahead(words, j, cities):
                break                                # "... Dr Merritt Island FL": the city begins
            for tok in toks:
                if tok in _NOT_STREET_WORDS:
                    stop = True; break               # criteria begin
                numbered_route = bool(name) and (
                    name[-1] == "hwy"
                    or (len(name) >= 2 and name[-1] == "rd" and name[-2] in ("county", "state"))
                )
                if tok.isdigit() and not numbered_route:
                    stop = True; break               # "123 Main St 3 bedroom"
                if tok in _DIRECTIONS and best is not None and best[1] == len(name):
                    stop = True; break               # trailing direction ends the street
                name.append(tok)
                if tok in _SUFFIX_SET and len(name) >= 2:
                    best = (j, len(name))
                if numbered_route and _ROUTE_DESIGNATOR.match(tok):
                    best = (j, len(name))            # "Highway A1a", "County Road 769"
            if stop or w.endswith(","):
                break
        if best is None:
            continue
        best, street = best[0], name[: best[1]]
        end_idx = best
        if best + 1 < len(words):                    # trailing direction: "... Rd SW"
            d = _tokens(words[best + 1][0])
            if (len(d) == 1 and d[0] in _DIRECTIONS and not words[best][0].endswith(",")
                    and not _city_ahead(words, best + 1, cities)):   # not "... Dr West Melbourne"
                street.append(d[0])
                end_idx = best + 1
        end = words[end_idx][2]
        if words[end_idx][0].endswith(","):
            end -= 1                                 # leave the comma for ", City, FL"
        tail = _TAIL_RE.match(query[end:])
        unit = city = state = zip_code = None
        if tail:
            unit = (tail.group("unit") or "").lower() or None
            city = (tail.group("city") or "").strip() or None
            state = (tail.group("state") or "").upper()[:2] or None
            zip_code = tail.group("zip")
            tail_end = end + tail.end()
            if city and any(t in _NOT_STREET_WORDS for t in city.lower().split()):
                # "1845 Hidden Lake Dr, with a pool": criteria, not a city — give
                # the text back to the remainder instead of swallowing it.
                tail_end = end + tail.start("city")
                city = None
                state = zip_code = None
            elif tail.group("city2"):
                # No comma: only a KNOWN city counts ("... Dr Rockledge", "... Dr
                # West Melbourne FL"); "... Dr granite countertops" is criteria.
                # Keep the longest known-city prefix, return the rest to the query.
                c2_start = end + tail.start("city2")
                c2_words = list(re.finditer(r"\S+", tail.group("city2")))
                acc: list[str] = []
                cut = None                       # (word count, char offset after it)
                for k, wm in enumerate(c2_words, 1):
                    acc.extend(_tokens(wm.group(0)))
                    if tuple(acc) in cities:
                        cut = (k, wm.end())
                if cut is None:
                    city = None
                    state = zip_code = None
                    tail_end = c2_start
                else:
                    city = tail.group("city2")[: cut[1]].strip()
                    if cut[0] < len(c2_words):   # criteria followed the city
                        state = zip_code = None
                        tail_end = c2_start + cut[1]
            end = tail_end
        remainder = (query[:s] + " " + query[end:]).strip(" ,.;")
        return AddressQuery(
            number=raw.strip(",.;").lower(), street=street, unit=unit, city=city,
            state=state, zip=zip_code, remainder=re.sub(r"\s{2,}", " ", remainder),
            matched_text=query[s:end],
        )
    return None


def _split_stored(street: str) -> tuple[str, list[str], str | None]:
    """'905 N Harbor City Blvd APT 106' -> ('905', ['n','harbor','city','blvd'], '106')."""
    toks = _tokens(street)
    if not toks:
        return "", [], None
    number, rest = toks[0], toks[1:]
    unit = None
    for i, t in enumerate(rest):
        if t in _UNIT_MARKERS:
            unit = " ".join(rest[i + 1:]) or None
            rest = rest[:i]
            break
    return number, rest, unit


def _street_matches(query_street: list[str], stored_street: list[str]) -> bool:
    if query_street == stored_street:
        return True
    # The user may omit a direction ("442 Lagrange St" for "442 Lagrange St SW",
    # "905 Harbor City Blvd" for "905 N Harbor City Blvd"); that still means the
    # same street. A DIFFERENT direction ("... St NE" vs "... St SW") does not —
    # if both sides carry directions they must agree exactly.
    q_dirs = [t for t in query_street if t in _DIRECTIONS]
    s_dirs = [t for t in stored_street if t in _DIRECTIONS]
    # Palm Bay streets carry both ("971 SW Richmond Cir SW"): typing only the
    # leading one is still the same street; a conflicting one is not.
    if q_dirs and s_dirs and q_dirs != s_dirs and q_dirs != s_dirs[:-1]:
        return False
    q = [t for t in query_street if t not in _DIRECTIONS]
    s = [t for t in stored_street if t not in _DIRECTIONS]
    # "McRae" vs "Mc Rae": same street, different spacing.
    return q == s or "".join(q) == "".join(s)


async def match_address(pool: asyncpg.Pool, addr: AddressQuery) -> list[int]:
    """Property ids whose stored address is this house (all units when no unit given)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, street, city, postal_code
            FROM properties
            WHERE street ~* ('^\\s*' || $1 || '\\M')
            """,
            re.escape(addr.number),
        )
    exact: list[int] = []
    out: list[int] = []
    for r in rows:
        number, street, unit = _split_stored(r["street"] or "")
        if number != addr.number or not _street_matches(addr.street, street):
            continue
        is_exact = street == addr.street
        if addr.unit:
            q_unit = re.sub(r"[\s-]", "", addr.unit)
            s_unit = re.sub(r"[\s-]", "", unit or "")
            if not s_unit or not (s_unit == q_unit or s_unit.startswith(q_unit)):
                continue
        # Disambiguate by ZIP / city only when the row actually carries one.
        if addr.zip and r["postal_code"] and r["postal_code"] != addr.zip:
            continue
        if addr.city and r["city"] and r["city"].strip().lower() != addr.city.lower():
            continue
        out.append(r["id"])
        if is_exact:
            exact.append(r["id"])
    # "971 SW Richmond Cir SW" and "971 Richmond Cir SW" are different listings:
    # an address typed exactly as stored wins over direction-tolerant near-matches;
    # an ambiguous form ("971 SW Richmond Cir") honestly returns both.
    return exact if exact else out
