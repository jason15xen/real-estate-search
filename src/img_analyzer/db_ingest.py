"""Database ingestion: convert processed Zillow data into the PostgreSQL schema."""

import logging
from collections import defaultdict

import asyncpg

logger = logging.getLogger(__name__)

# resoFacts.rooms roomType → our room types
RESO_ROOM_MAP = {
    "MasterBedroom": "Bedroom",
    "Bedroom": "Bedroom",
    "MasterBathroom": "Bathroom",
    "Bathroom": "Bathroom",
    "Kitchen": "Kitchen",
    "DiningRoom": "Dining Room",
    "LivingRoom": "Living Room",
    "FamilyRoom": "Living Room",
    "Garage": "Garage",
}

# room types → DB columns
ROOM_COUNT_COLUMNS = {
    "Bedroom": "bedroom_count",
    "Bathroom": "bathroom_count",
    "Kitchen": "kitchen_count",
    "Living Room": "living_room_count",
    "Dining Room": "dining_room_count",
    "Garage": "garage_count",
}


def _normalize_listing_terms(raw: str | None) -> list[str]:
    """'Cash,Conventional,VA Loan' -> ['cash','conventional','va_loan']."""
    if not raw or not isinstance(raw, str):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for term in raw.split(","):
        norm = "_".join(term.strip().lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _canonical_photo_url(photo: dict) -> str | None:
    """Highest-res JPEG URL for a photo; canonical id for diffing and room_instances.photo_url."""
    jpegs = ((photo.get("mixedSources") or {}).get("jpeg") or [])
    if not jpegs:
        return None
    best = max(jpegs, key=lambda j: j.get("width", 0))
    return best.get("url") or None


def _build_rooms_from_photos(photos: list[dict]) -> dict[str, list[dict]]:
    """Group features by RoomType → {room_type: [{"features","color","photo_url"}]}; unusable results become empty "Unknown" stubs to claim their URL."""
    rooms: dict[str, list[dict]] = defaultdict(list)
    for photo in photos:
        room_type = photo.get("RoomType", "Unknown")
        features = photo.get("Features", [])
        color = photo.get("Color")
        if isinstance(color, str):
            color = color.strip().lower() or None
            if color in {"unknown", "n/a", "none", "null"}:
                color = None
        photo_url = _canonical_photo_url(photo)

        if room_type and room_type != "Unknown" and features:
            rooms[room_type].append({
                "features": features,
                "color": color,
                "photo_url": photo_url,
            })
        elif photo_url:
            # Unusable Vision result: claim URL as stub to avoid infinite re-analyze.
            rooms["Unknown"].append({
                "features": [],
                "color": None,
                "photo_url": photo_url,
            })
    return dict(rooms)


def _get_room_counts(record: dict, room_counts_by_type: dict[str, int]) -> dict[str, int]:
    """Compute the 6 denormalized room-count columns; Bedroom/Bathroom from Zillow, others from resoFacts.rooms > room_counts_by_type > hasGarage capacity."""
    counts: dict[str, int] = {
        "Bedroom": int(record.get("bedrooms") or 0),
        "Bathroom": int(record.get("bathrooms") or 0),
    }

    reso_facts = record.get("resoFacts", {}) or {}
    reso_rooms = reso_facts.get("rooms", []) or []
    reso_counts: dict[str, int] = defaultdict(int)
    for r in reso_rooms:
        raw_type = r.get("roomType", "")
        mapped = RESO_ROOM_MAP.get(raw_type)
        if mapped:
            reso_counts[mapped] += 1

    for room_type in ["Kitchen", "Dining Room", "Living Room", "Garage"]:
        counts[room_type] = reso_counts.get(room_type, 0)

    # Fallback to provided source when resoFacts had no count.
    for room_type, n in room_counts_by_type.items():
        if room_type in ROOM_COUNT_COLUMNS and counts.get(room_type, 0) == 0:
            counts[room_type] = n

    # Garage fallback from hasGarage flag when no other count exists.
    if counts.get("Garage", 0) == 0:
        if reso_facts.get("hasGarage") or reso_facts.get("hasAttachedGarage"):
            capacity = reso_facts.get("garageParkingCapacity", 1) or 1
            counts["Garage"] = capacity

    return counts


async def query_room_instance_counts(conn, property_id: int) -> dict[str, int]:
    """{room_type: count} for a property's room_instances, excluding 'Unknown' stubs."""
    rows = await conn.fetch(
        """
        SELECT room_type, COUNT(*) AS n FROM room_instances
        WHERE property_id = $1 AND room_type != 'Unknown'
        GROUP BY room_type
        """,
        property_id,
    )
    return {r["room_type"]: r["n"] for r in rows}


async def refresh_property_room_counts(conn, existing_id: int, item: dict) -> None:
    """Recompute Kitchen/Living/Dining/Garage from current room_instances; call after update_property_scalars."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    current_counts = await query_room_instance_counts(conn, existing_id)
    counts = _get_room_counts(record, current_counts)
    await conn.execute(
        """
        UPDATE properties SET
            kitchen_count = $2,
            living_room_count = $3,
            dining_room_count = $4,
            garage_count = $5,
            updated_at = NOW()
        WHERE id = $1
        """,
        existing_id,
        counts.get("Kitchen", 0),
        counts.get("Living Room", 0),
        counts.get("Dining Room", 0),
        counts.get("Garage", 0),
    )


# Helpers for single-property create/update (POST/PUT /properties)

def _extract_neighborhood(record: dict) -> str | None:
    """Zillow neighborhood name from neighborhoodSearchUrl, resolved against nearbyNeighborhoods or de-slugged from the URL path; None when not in a named neighborhood."""
    u = record.get("neighborhoodSearchUrl") or {}
    path = u.get("path") if isinstance(u, dict) else None
    if not path:
        return None
    for nn in (record.get("nearbyNeighborhoods") or []):
        if isinstance(nn, dict) and (nn.get("regionUrl") or {}).get("path") == path and nn.get("name"):
            return nn["name"]
    # fallback: "/viera-east-melbourne-fl/" -> "Viera East"
    slug = path.strip("/")
    slug = slug[:-3] if slug.endswith("-fl") else slug.split("-fl/")[0]
    parts = slug.split("-")
    cities = {"melbourne", "titusville", "rockledge", "mims"}
    while parts and parts[-1].lower() in cities:
        parts.pop()
    return " ".join(p.capitalize() for p in parts) or None


def _extract_locality(record: dict) -> str | None:
    """OSM/Photon place name from the PhotonPropertyFullAddress reverse-geocode in the record; None when absent."""
    ph = record.get("PhotonPropertyFullAddress") or {}
    feats = ph.get("features") or []
    if feats and isinstance(feats[0], dict):
        return (feats[0].get("properties") or {}).get("city") or None
    return None


def _extract_property_fields(item: dict) -> dict:
    """Extract DB-relevant scalar fields from a Zillow item, keyed like properties columns."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    address = record.get("address", {}) or {}
    reso_facts = record.get("resoFacts", {}) or {}

    lot_size = record.get("lotSize", 0) or 0
    lot_units = record.get("lotAreaUnits", "")
    if lot_units == "Acres":
        lot_size = int(float(record.get("lotAreaValue", 0) or 0) * 43560)
    else:
        lot_size = int(lot_size)

    return {
        "guid": item.get("Id", ""),
        "name": address.get("streetAddress", "Unknown Property"),
        "street": address.get("streetAddress", ""),
        "district": address.get("subdivision", ""),
        "city": address.get("city", ""),
        "state": address.get("state", ""),
        "postal_code": address.get("zipcode", ""),
        "country": "US",
        "county": (record.get("county") or "").strip() or None,
        "locality": _extract_locality(record),
        "neighborhood": _extract_neighborhood(record),
        "latitude": float(record.get("latitude", 0) or 0),
        "longitude": float(record.get("longitude", 0) or 0),
        "area_sqft": int(record.get("livingArea", 0) or 0),
        "price_usd": int(record.get("price", 0) or 0),
        "home_type": record.get("homeType"),
        "rent_estimate": record.get("rentZestimate"),
        "year_built": record.get("yearBuilt"),
        "lot_size_sqft": lot_size,
        "stories": reso_facts.get("stories"),
        "has_pool": bool(reso_facts.get("hasPrivatePool")),
        "has_waterfront": bool(reso_facts.get("hasWaterfrontView")),
        "description": record.get("description"),
        "financing": _normalize_listing_terms(reso_facts.get("listingTerms")),
    }


def _extract_schools(item: dict) -> list[dict]:
    """Normalized schools list from incoming item."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    out: list[dict] = []
    for s in record.get("schools", []) or []:
        out.append({
            "name": s.get("name", ""),
            "rating": s.get("rating"),
            "grades": s.get("grades", ""),
            "distance": float(s.get("distance", 0) or 0),
            "link": s.get("link", ""),
        })
    return out


# Feature tags meaning the pool water itself is roofed/screened/caged; used to derive properties.has_covered_pool.
_COVERED_POOL_TAGS = [
    "covered pool", "screened pool", "screened-in pool", "screen-enclosed pool",
    "screen enclosed pool", "enclosed pool", "caged pool", "pool cage",
    "covered pool cage", "pool cage enclosure", "pool enclosure",
    "screened pool cage", "screened pool enclosure", "screened pool enclosures",
]


async def _refresh_has_covered_pool(conn, prop_id: int) -> None:
    """Recompute properties.has_covered_pool from current room instances: TRUE iff any covered-pool tag is present; idempotent."""
    await conn.execute(
        """
        UPDATE properties p SET has_covered_pool = EXISTS (
            SELECT 1 FROM room_instances ri
            WHERE ri.property_id = p.id AND ri.features && $2::text[]
        )
        WHERE p.id = $1
        """,
        prop_id, _COVERED_POOL_TAGS,
    )


async def _insert_children(conn, prop_id: int, rooms_from_photos: dict, schools: list[dict]) -> None:
    """Insert rooms, room_instances, and property_schools for a property."""
    for room_type, instances in rooms_from_photos.items():
        room_id = await conn.fetchval("""
            INSERT INTO rooms (property_id, room_type, count)
            VALUES ($1, $2, $3) RETURNING id
        """, prop_id, room_type, len(instances))
        for idx, inst in enumerate(instances):
            features = inst["features"]
            color = inst.get("color")
            photo_url = inst.get("photo_url")
            features_text = ", ".join(features)
            await conn.execute("""
                INSERT INTO room_instances (
                    room_id, property_id, room_type,
                    instance_index, features, features_text, color, photo_url
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, room_id, prop_id, room_type, idx, features, features_text, color, photo_url)
    for s in schools:
        await conn.execute("""
            INSERT INTO property_schools (
                property_id, school_name, rating, grades, distance_miles, link
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """, prop_id, s["name"], s["rating"], s["grades"], s["distance"], s["link"])
    # Derive has_covered_pool from the room_instances just written.
    await _refresh_has_covered_pool(conn, prop_id)


async def update_property_scalars(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Update non-photo-derived columns: scalars plus Bedroom/Bathroom counts from Zillow."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    await conn.execute("""
        UPDATE properties SET
            name=$2, street=$3, district=$4, city=$5, state=$6,
            postal_code=$7, country=$8,
            geom=ST_MakePoint($9, $10)::geography,
            area_sqft=$11, price_usd=$12,
            bedroom_count=$13, bathroom_count=$14,
            home_type=$15, rent_estimate=$16, year_built=$17,
            lot_size_sqft=$18, stories=$19,
            has_pool=$20, has_waterfront=$21, description=$22, financing=$23,
            county=$24, locality=$25, neighborhood=$26,
            updated_at=NOW()
        WHERE id = $1
    """,
        existing_id,
        fields["name"], fields["street"], fields["district"], fields["city"],
        fields["state"], fields["postal_code"], fields["country"],
        fields["longitude"], fields["latitude"],
        fields["area_sqft"], fields["price_usd"],
        int(record.get("bedrooms") or 0),
        int(record.get("bathrooms") or 0),
        fields["home_type"], fields["rent_estimate"], fields["year_built"],
        fields["lot_size_sqft"], fields["stories"],
        fields["has_pool"], fields["has_waterfront"], fields["description"],
        fields["financing"],
        fields["county"], fields["locality"], fields["neighborhood"],
    )
    # Re-derive has_covered_pool (room features may have changed).
    await _refresh_has_covered_pool(conn, existing_id)


async def update_property_metadata(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Update only the properties row (scalars + room counts); no child rows."""
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    room_counts = _get_room_counts(
        record, {rt: len(insts) for rt, insts in rooms_from_photos.items()}
    )
    await conn.execute("""
        UPDATE properties SET
            name=$2, street=$3, district=$4, city=$5, state=$6,
            postal_code=$7, country=$8,
            geom=ST_MakePoint($9, $10)::geography,
            area_sqft=$11, price_usd=$12,
            bedroom_count=$13, bathroom_count=$14, kitchen_count=$15,
            living_room_count=$16, dining_room_count=$17, garage_count=$18,
            home_type=$19, rent_estimate=$20, year_built=$21,
            lot_size_sqft=$22, stories=$23,
            has_pool=$24, has_waterfront=$25, description=$26, financing=$27,
            county=$28, locality=$29, neighborhood=$30,
            updated_at=NOW()
        WHERE id = $1
    """,
        existing_id,
        fields["name"], fields["street"], fields["district"], fields["city"],
        fields["state"], fields["postal_code"], fields["country"],
        fields["longitude"], fields["latitude"],
        fields["area_sqft"], fields["price_usd"],
        room_counts.get("Bedroom", 0), room_counts.get("Bathroom", 0),
        room_counts.get("Kitchen", 0), room_counts.get("Living Room", 0),
        room_counts.get("Dining Room", 0), room_counts.get("Garage", 0),
        fields["home_type"], fields["rent_estimate"], fields["year_built"],
        fields["lot_size_sqft"], fields["stories"],
        fields["has_pool"], fields["has_waterfront"], fields["description"],
        fields["financing"],
        fields["county"], fields["locality"], fields["neighborhood"],
    )
    # Re-derive has_covered_pool (room features may have changed).
    await _refresh_has_covered_pool(conn, existing_id)


async def update_property_with_children(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Full re-process: update properties row and replace all child rows. Used when photos/schools changed."""
    await conn.execute("DELETE FROM property_schools WHERE property_id = $1", existing_id)
    await conn.execute("DELETE FROM room_instances WHERE property_id = $1", existing_id)
    await conn.execute("DELETE FROM rooms WHERE property_id = $1", existing_id)
    await update_property_metadata(conn, existing_id, item)

    record = item.get("ZillowPropertyRecord", {}) or {}
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    await _insert_children(conn, existing_id, rooms_from_photos, _extract_schools(item))

