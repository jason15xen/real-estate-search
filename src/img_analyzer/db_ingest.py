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

import json
import logging
from collections import defaultdict
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

PROCESSED_FILE = Path(__file__).resolve().parent.parent / "processed" / "data.json"

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


def _build_rooms_from_photos(photos: list[dict]) -> dict[str, list[list[str]]]:
    """
    Group extracted features by RoomType from processed photos.
    Returns: { room_type: [[features_photo_1], [features_photo_2], ...] }
    """
    rooms: dict[str, list[list[str]]] = defaultdict(list)
    for photo in photos:
        room_type = photo.get("RoomType", "Unknown")
        features = photo.get("Features", [])
        if room_type and room_type != "Unknown" and features:
            rooms[room_type].append(features)
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


async def ingest_processed_data(pool: asyncpg.Pool) -> dict[str, int]:
    """
    Read src/processed/data.json, transform to DB schema, and insert.
    Returns stats: { total_properties, total_rooms, total_room_instances }
    """
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_FILE}")

    with open(PROCESSED_FILE, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("Processed data file is empty")

    stats = {
        "total_properties": 0, "updated_properties": 0,
        "total_rooms": 0, "total_room_instances": 0, "total_schools": 0,
    }

    async with pool.acquire() as conn:
        for item in data:
            record = item.get("ZillowPropertyRecord", {})
            address = record.get("address", {})
            photos = record.get("originalPhotos", [])

            # Build rooms from extracted photo features
            rooms_from_photos = _build_rooms_from_photos(photos)

            # Determine room counts
            room_counts = _get_room_counts(record, rooms_from_photos)

            # Property name: use street address
            name = address.get("streetAddress", "Unknown Property")

            lat = record.get("latitude", 0) or 0
            lng = record.get("longitude", 0) or 0
            area = record.get("livingArea", 0) or 0
            price = record.get("price", 0) or 0
            reso_facts = record.get("resoFacts", {}) or {}

            # New fields
            home_type = record.get("homeType")
            rent_estimate = record.get("rentZestimate")
            year_built = record.get("yearBuilt")
            lot_size = record.get("lotSize", 0) or 0
            lot_units = record.get("lotAreaUnits", "")
            if lot_units == "Acres":
                lot_size = int(float(record.get("lotAreaValue", 0) or 0) * 43560)
            else:
                lot_size = int(lot_size)
            stories_val = reso_facts.get("stories")
            has_pool = bool(reso_facts.get("hasPrivatePool"))
            has_waterfront = bool(reso_facts.get("hasWaterfrontView"))
            description = record.get("description")
            financing = _normalize_listing_terms(reso_facts.get("listingTerms"))

            # Original GUID from data.json
            guid = item.get("Id", "")

            # Check if property already exists
            existing_id = await conn.fetchval(
                "SELECT id FROM properties WHERE guid = $1", guid
            )

            if existing_id:
                # Update existing — delete old child data (CASCADE doesn't auto-trigger on UPDATE)
                await conn.execute("DELETE FROM property_schools WHERE property_id = $1", existing_id)
                await conn.execute("DELETE FROM room_instances WHERE property_id = $1", existing_id)
                await conn.execute("DELETE FROM rooms WHERE property_id = $1", existing_id)

                # Update property row
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
                        has_pool=$24, has_waterfront=$25, description=$26,
                        financing=$27,
                        updated_at=NOW()
                    WHERE id = $1
                """,
                    existing_id,
                    name,
                    address.get("streetAddress", ""),
                    address.get("subdivision", ""),
                    address.get("city", ""),
                    address.get("state", ""),
                    address.get("zipcode", ""),
                    "US",
                    lng, lat,
                    int(area), int(price),
                    room_counts.get("Bedroom", 0),
                    room_counts.get("Bathroom", 0),
                    room_counts.get("Kitchen", 0),
                    room_counts.get("Living Room", 0),
                    room_counts.get("Dining Room", 0),
                    room_counts.get("Garage", 0),
                    home_type, rent_estimate, year_built,
                    lot_size, stories_val,
                    has_pool, has_waterfront, description,
                    financing,
                )
                prop_id = existing_id
                stats["updated_properties"] += 1
            else:
                # Insert new property
                prop_id = await conn.fetchval("""
                    INSERT INTO properties (
                        guid, name, street, district, city, state, postal_code, country,
                        geom, area_sqft, price_usd,
                        bedroom_count, bathroom_count, kitchen_count,
                        living_room_count, dining_room_count, garage_count,
                        home_type, rent_estimate, year_built,
                        lot_size_sqft, stories,
                        has_pool, has_waterfront, description,
                        financing
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8,
                        ST_MakePoint($9, $10)::geography,
                        $11, $12, $13, $14, $15, $16, $17, $18,
                        $19, $20, $21, $22, $23, $24, $25, $26,
                        $27
                    ) RETURNING id
                """,
                    guid,
                    name,
                    address.get("streetAddress", ""),
                    address.get("subdivision", ""),
                    address.get("city", ""),
                    address.get("state", ""),
                    address.get("zipcode", ""),
                    "US",
                    lng, lat,
                    int(area), int(price),
                    room_counts.get("Bedroom", 0),
                    room_counts.get("Bathroom", 0),
                    room_counts.get("Kitchen", 0),
                    room_counts.get("Living Room", 0),
                    room_counts.get("Dining Room", 0),
                    room_counts.get("Garage", 0),
                    home_type, rent_estimate, year_built,
                    lot_size, stories_val,
                    has_pool, has_waterfront, description,
                    financing,
                )

            stats["total_properties"] += 1

            # Insert rooms and room_instances from photo-extracted features
            for room_type, instances in rooms_from_photos.items():
                room_id = await conn.fetchval("""
                    INSERT INTO rooms (property_id, room_type, count)
                    VALUES ($1, $2, $3) RETURNING id
                """, prop_id, room_type, len(instances))

                stats["total_rooms"] += 1

                for idx, features in enumerate(instances):
                    features_text = ", ".join(features)
                    await conn.execute("""
                        INSERT INTO room_instances (
                            room_id, property_id, room_type,
                            instance_index, features, features_text
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        room_id, prop_id, room_type,
                        idx, features, features_text,
                    )
                    stats["total_room_instances"] += 1

            # Insert schools
            schools = record.get("schools", []) or []
            for school in schools:
                await conn.execute("""
                    INSERT INTO property_schools (
                        property_id, school_name, rating, grades,
                        distance_miles, link
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    prop_id,
                    school.get("name", ""),
                    school.get("rating"),
                    school.get("grades", ""),
                    school.get("distance", 0),
                    school.get("link", ""),
                )
                stats["total_schools"] += 1

    new_count = stats['total_properties'] - stats['updated_properties']
    logger.info(
        f"Ingested {stats['total_properties']} properties "
        f"({new_count} new, {stats['updated_properties']} updated), "
        f"{stats['total_rooms']} rooms, "
        f"{stats['total_room_instances']} room instances, "
        f"{stats['total_schools']} schools"
    )
    return stats
