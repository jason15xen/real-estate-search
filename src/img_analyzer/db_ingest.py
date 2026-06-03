"""
Database Ingestion — Converts processed Zillow data (with extracted features)
into the PostgreSQL schema used by the search pipeline.

Mapping:
  Zillow data                    → DB Schema
  ─────────────────────────────────────────────────────
  address.streetAddress          → properties.street
  address.subdivision            → properties.district
  address.city                   → properties.city
  address.state                  → properties.state
  address.zipcode                → properties.postal_code
  "US"                           → properties.country
  longitude, latitude            → properties.geom (PostGIS)
  livingArea                     → properties.area_sqft
  price                          → properties.price_usd
  bedrooms                       → properties.bedroom_count
  bathrooms                      → properties.bathroom_count
  resoFacts.rooms                → room counts (kitchen, dining, etc.)

  originalPhotos[].RoomType      → room_instances.room_type
  originalPhotos[].Features      → room_instances.features / features_text
"""

import logging
from collections import defaultdict

import asyncpg

logger = logging.getLogger(__name__)

# Map resoFacts.rooms roomType values to our room types
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

# Map our room types to DB columns
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


def _build_rooms_from_photos(photos: list[dict]) -> dict[str, list[dict]]:
    """
    Group extracted features by RoomType from processed photos.
    Returns: { room_type: [
        {"features": [...], "color": "white" | None},
        ...
    ]}
    """
    rooms: dict[str, list[dict]] = defaultdict(list)
    for photo in photos:
        room_type = photo.get("RoomType", "Unknown")
        features = photo.get("Features", [])
        color = photo.get("Color")
        if isinstance(color, str):
            color = color.strip().lower() or None
            if color in {"unknown", "n/a", "none", "null"}:
                color = None
        if room_type and room_type != "Unknown" and features:
            rooms[room_type].append({"features": features, "color": color})
    return dict(rooms)


def _get_room_counts(record: dict, rooms_from_photos: dict[str, list]) -> dict[str, int]:
    """
    Determine room counts from Zillow data fields + photo analysis.
    Priority: Zillow structured data > photo-derived counts.
    """
    counts: dict[str, int] = {
        "Bedroom": record.get("bedrooms", 0) or 0,
        "Bathroom": record.get("bathrooms", 0) or 0,
    }

    # Check resoFacts.rooms for additional room types
    reso_facts = record.get("resoFacts", {}) or {}
    reso_rooms = reso_facts.get("rooms", []) or []
    reso_counts: dict[str, int] = defaultdict(int)
    for r in reso_rooms:
        raw_type = r.get("roomType", "")
        mapped = RESO_ROOM_MAP.get(raw_type)
        if mapped:
            reso_counts[mapped] += 1

    # Use resoFacts counts for non-bed/bath room types
    for room_type in ["Kitchen", "Dining Room", "Living Room", "Garage"]:
        counts[room_type] = reso_counts.get(room_type, 0)

    # If resoFacts has no kitchen but photos found one, use photo count
    for room_type, instances in rooms_from_photos.items():
        if room_type in ROOM_COUNT_COLUMNS and counts.get(room_type, 0) == 0:
            counts[room_type] = len(instances)

    # Check garage from resoFacts flags
    if counts.get("Garage", 0) == 0:
        if reso_facts.get("hasGarage") or reso_facts.get("hasAttachedGarage"):
            capacity = reso_facts.get("garageParkingCapacity", 1) or 1
            counts["Garage"] = capacity

    return counts


# ===================================================================
# Helpers for single-property create/update (used by POST/PUT /properties)
# ===================================================================

def _extract_property_fields(item: dict) -> dict:
    """Extract all DB-relevant scalar fields from a Zillow property dict.

    Returns a normalized dict ready for INSERT/UPDATE comparison. The dict
    contains the same key-value shape as the properties table columns.
    """
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


def _extract_photo_urls(item: dict) -> list[str]:
    """Highest-resolution JPEG URL for each photo in originalPhotos. Used to
    detect whether the photo set has changed between incoming and current data.
    """
    record = item.get("ZillowPropertyRecord", {}) or {}
    urls: list[str] = []
    for photo in record.get("originalPhotos", []) or []:
        jpegs = (photo.get("mixedSources", {}) or {}).get("jpeg", []) or []
        if not jpegs:
            continue
        best = max(jpegs, key=lambda j: j.get("width", 0))
        urls.append(best.get("url", ""))
    return urls


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


async def _load_existing_property(conn, guid: str) -> dict | None:
    """Fetch current DB state for a property (scalar fields + schools).
    Returns None if the property doesn't exist.
    """
    row = await conn.fetchrow("""
        SELECT id, guid, name, street, district, city, state, postal_code, country,
               ST_Y(geom::geometry) AS latitude,
               ST_X(geom::geometry) AS longitude,
               area_sqft, price_usd,
               bedroom_count, bathroom_count, kitchen_count,
               living_room_count, dining_room_count, garage_count,
               home_type, rent_estimate, year_built,
               lot_size_sqft, stories,
               has_pool, has_waterfront, description, financing
        FROM properties WHERE guid = $1
    """, guid)
    if not row:
        return None
    schools = await conn.fetch("""
        SELECT school_name, rating, grades, distance_miles, link
        FROM property_schools WHERE property_id = $1 ORDER BY distance_miles, school_name
    """, row["id"])
    return {
        "id": row["id"],
        "scalars": {
            "guid": row["guid"],
            "name": row["name"],
            "street": row["street"],
            "district": row["district"],
            "city": row["city"],
            "state": row["state"],
            "postal_code": row["postal_code"],
            "country": row["country"],
            "latitude": float(row["latitude"]) if row["latitude"] is not None else 0.0,
            "longitude": float(row["longitude"]) if row["longitude"] is not None else 0.0,
            "area_sqft": row["area_sqft"],
            "price_usd": row["price_usd"],
            "home_type": row["home_type"],
            "rent_estimate": row["rent_estimate"],
            "year_built": row["year_built"],
            "lot_size_sqft": row["lot_size_sqft"],
            "stories": row["stories"],
            "has_pool": row["has_pool"],
            "has_waterfront": row["has_waterfront"],
            "description": row["description"],
            "financing": list(row["financing"]) if row["financing"] else [],
        },
        "room_counts": {
            "Bedroom": row["bedroom_count"],
            "Bathroom": row["bathroom_count"],
            "Kitchen": row["kitchen_count"],
            "Living Room": row["living_room_count"],
            "Dining Room": row["dining_room_count"],
            "Garage": row["garage_count"],
        },
        "schools": [
            {
                "name": s["school_name"],
                "rating": s["rating"],
                "grades": s["grades"],
                "distance": float(s["distance_miles"]) if s["distance_miles"] is not None else 0.0,
                "link": s["link"],
            }
            for s in schools
        ],
    }


def _diff_scalar_fields(incoming: dict, current: dict) -> list[str]:
    """Return the list of property-table fields whose values differ.
    Compares numeric fields with a small tolerance; everything else is strict.
    """
    changed: list[str] = []
    skip_keys = {"guid"}  # immutable identifier
    for key, new_val in incoming.items():
        if key in skip_keys:
            continue
        old_val = current.get(key)
        if key in {"latitude", "longitude"}:
            old_f = float(old_val) if old_val is not None else 0.0
            new_f = float(new_val) if new_val is not None else 0.0
            if abs(old_f - new_f) > 1e-7:
                changed.append(key)
        elif key == "financing":
            if sorted(old_val or []) != sorted(new_val or []):
                changed.append(key)
        else:
            if (old_val or None) != (new_val or None):
                changed.append(key)
    return changed


def _schools_changed(incoming: list[dict], current: list[dict]) -> bool:
    """True if the schools array materially differs (name/rating/distance)."""
    def key(s: dict) -> tuple:
        return (s.get("name", ""), s.get("rating"), round(float(s.get("distance") or 0), 2))
    return sorted(map(key, incoming)) != sorted(map(key, current))


async def insert_property(
    conn,
    item: dict,
) -> dict:
    """Insert ONE property + its children. Caller must verify the GUID doesn't
    already exist. Returns stats dict.
    """
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    room_counts = _get_room_counts(record, rooms_from_photos)

    prop_id = await conn.fetchval("""
        INSERT INTO properties (
            guid, name, street, district, city, state, postal_code, country,
            geom, area_sqft, price_usd,
            bedroom_count, bathroom_count, kitchen_count,
            living_room_count, dining_room_count, garage_count,
            home_type, rent_estimate, year_built,
            lot_size_sqft, stories,
            has_pool, has_waterfront, description, financing
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8,
            ST_MakePoint($9, $10)::geography,
            $11, $12, $13, $14, $15, $16, $17, $18,
            $19, $20, $21, $22, $23, $24, $25, $26, $27
        ) RETURNING id
    """,
        fields["guid"], fields["name"], fields["street"], fields["district"],
        fields["city"], fields["state"], fields["postal_code"], fields["country"],
        fields["longitude"], fields["latitude"],
        fields["area_sqft"], fields["price_usd"],
        room_counts.get("Bedroom", 0), room_counts.get("Bathroom", 0),
        room_counts.get("Kitchen", 0), room_counts.get("Living Room", 0),
        room_counts.get("Dining Room", 0), room_counts.get("Garage", 0),
        fields["home_type"], fields["rent_estimate"], fields["year_built"],
        fields["lot_size_sqft"], fields["stories"],
        fields["has_pool"], fields["has_waterfront"], fields["description"],
        fields["financing"],
    )

    await _insert_children(conn, prop_id, rooms_from_photos, _extract_schools(item))
    return {"prop_id": prop_id}


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
            features_text = ", ".join(features)
            await conn.execute("""
                INSERT INTO room_instances (
                    room_id, property_id, room_type,
                    instance_index, features, features_text, color
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, room_id, prop_id, room_type, idx, features, features_text, color)
    for s in schools:
        await conn.execute("""
            INSERT INTO property_schools (
                property_id, school_name, rating, grades, distance_miles, link
            ) VALUES ($1, $2, $3, $4, $5, $6)
        """, prop_id, s["name"], s["rating"], s["grades"], s["distance"], s["link"])


async def update_property_metadata(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Update ONLY the properties row (scalar fields + room counts).
    No changes to room_instances or property_schools.
    """
    record = item.get("ZillowPropertyRecord", {}) or {}
    fields = _extract_property_fields(item)
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    room_counts = _get_room_counts(record, rooms_from_photos)
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
    )


async def update_property_with_children(
    conn,
    existing_id: int,
    item: dict,
) -> None:
    """Full re-process: update properties row AND replace all child rows
    (rooms, room_instances, property_schools). Used when photos or schools changed.
    """
    await conn.execute("DELETE FROM property_schools WHERE property_id = $1", existing_id)
    await conn.execute("DELETE FROM room_instances WHERE property_id = $1", existing_id)
    await conn.execute("DELETE FROM rooms WHERE property_id = $1", existing_id)
    await update_property_metadata(conn, existing_id, item)

    record = item.get("ZillowPropertyRecord", {}) or {}
    rooms_from_photos = _build_rooms_from_photos(record.get("originalPhotos", []) or [])
    await _insert_children(conn, existing_id, rooms_from_photos, _extract_schools(item))

