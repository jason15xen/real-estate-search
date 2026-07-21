"""US state / country name normalization, shared by search filtering and ingest
enrichment. Users type "Florida" while records store "FL" (and Photon returns
"Florida") — matching must accept either form."""

from __future__ import annotations

STATE_ABBREV: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
_ABBREV_TO_NAME = {code.lower(): name for name, code in STATE_ABBREV.items()}

_COUNTRY_GROUPS = [{"us", "usa", "u.s.", "u.s.a.", "united states", "united states of america"}]


def abbrev_state(name: str) -> str:
    """'Florida' -> 'FL' (the record/DB convention); short or unknown values pass through."""
    cleaned = (name or "").strip()
    return STATE_ABBREV.get(cleaned.lower(), cleaned) if len(cleaned) > 2 else cleaned


def state_variants(value: str) -> list[str]:
    """All lowercase forms a stored state column might use for this input:
    'Florida' -> ['florida', 'fl']; 'fl' -> ['fl', 'florida']. Empty input -> []."""
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return []
    variants = {cleaned}
    if cleaned in STATE_ABBREV:                 # full name given
        variants.add(STATE_ABBREV[cleaned].lower())
    if cleaned in _ABBREV_TO_NAME:              # abbreviation given
        variants.add(_ABBREV_TO_NAME[cleaned])
    return sorted(variants)


def country_variants(value: str) -> list[str]:
    """Accepted forms for a country value ('United States' <-> 'US')."""
    cleaned = (value or "").strip().lower()
    if not cleaned:
        return []
    for group in _COUNTRY_GROUPS:
        if cleaned in group:
            return sorted(group)
    return [cleaned]


def expand_state(value: str) -> str:
    """'FL' -> 'Florida' (title-cased full name); full names and unknown values pass
    through unchanged. Used where a human-geocodable place name is needed."""
    cleaned = (value or "").strip()
    full = _ABBREV_TO_NAME.get(cleaned.lower())
    return full.title() if full else cleaned
