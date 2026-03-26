"""
Data Ingestion — Loads property data from mockup.json into PostgreSQL.
"""

import asyncio
import json
import logging

import asyncpg

logger = logging.getLogger(__name__)


async def ingest_properties(
    db_pool: asyncpg.Pool,
    file_path: str = "mockup.json",
) -> int:
    with open(file_path) as f:
        data = json.load(f)

    count = 0
    async with db_pool.acquire() as conn:
        # Clear existing data
        await conn.execute("DELETE FROM room_instances")
        await conn.execute("DELETE FROM rooms")
        await conn.execute("DELETE FROM properties")

        for item in data:
            addr = item["Address"]

            # Count rooms by type
            room_counts = {}
            for room in item["Rooms"]:
                room_counts[room["Type"]] = room["Count"]

            # Insert property
            prop_id = await conn.fetchval("""
                INSERT INTO properties (
                    name, street, district, city, state, postal_code, country,
                    geom, area_sqft, price_usd,
                    bedroom_count, bathroom_count, kitchen_count,
                    living_room_count, dining_room_count, garage_count
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7,
                    ST_MakePoint($8, $9)::geography,
                    $10, $11, $12, $13, $14, $15, $16, $17
                ) RETURNING id
            """,
                item["Name"],
                addr["Street"], addr["District"], addr["City"],
                addr["State"], addr["PostalCode"], addr["Country"],
                addr["Longitude"], addr["Latitude"],
                item["AreaSqft"], item["PriceUSD"],
                room_counts.get("Bedroom", 0),
                room_counts.get("Bathroom", 0),
                room_counts.get("Kitchen", 0),
                room_counts.get("Living Room", 0),
                room_counts.get("Dining Room", 0),
                room_counts.get("Garage", 0),
            )

            # Insert rooms and room instances
            for room in item["Rooms"]:
                room_id = await conn.fetchval("""
                    INSERT INTO rooms (property_id, room_type, count)
                    VALUES ($1, $2, $3) RETURNING id
                """, prop_id, room["Type"], room["Count"])

                for i, instance in enumerate(room["Instances"]):
                    features = instance["Features"]
                    features_text = ", ".join(features)
                    await conn.execute("""
                        INSERT INTO room_instances (
                            room_id, property_id, room_type,
                            instance_index, features, features_text
                        ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        room_id, prop_id, room["Type"],
                        i, features, features_text,
                    )

            count += 1

    logger.info(f"Ingested {count} properties into PostgreSQL")
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        import os
        pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "admin"),
            password=os.getenv("POSTGRES_PASSWORD", "admin123"),
            database=os.getenv("POSTGRES_DB", "real_estate"),
        )
        count = await ingest_properties(pool, "mockup.json")
        print(f"Done: {count} properties ingested")
        await pool.close()

    asyncio.run(main())
