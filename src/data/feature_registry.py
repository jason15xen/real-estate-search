"""
Feature Registry — Collects all unique feature names and room types from PostgreSQL
at startup and stores them in memory.

This allows the query parser to map user input to exact feature names,
eliminating the need for vector search and LLM validation.
"""

import logging

import asyncpg

logger = logging.getLogger(__name__)


class FeatureRegistry:
    def __init__(self):
        self.features: set[str] = set()
        self.room_types: set[str] = set()
        self.features_by_room_type: dict[str, set[str]] = {}

    async def build_from_db(self, pool: asyncpg.Pool) -> None:
        """Load unique features and room types from PostgreSQL."""
        async with pool.acquire() as conn:
            # Get all unique room types
            rows = await conn.fetch(
                "SELECT DISTINCT room_type FROM room_instances ORDER BY room_type"
            )
            for row in rows:
                self.room_types.add(row["room_type"])

            # Get all unique features per room type
            rows = await conn.fetch("""
                SELECT DISTINCT room_type, unnest(features) AS feature
                FROM room_instances
                ORDER BY room_type, feature
            """)
            for row in rows:
                feature = row["feature"]
                room_type = row["room_type"]
                self.features.add(feature)
                if room_type not in self.features_by_room_type:
                    self.features_by_room_type[room_type] = set()
                self.features_by_room_type[room_type].add(feature)

        logger.info(
            f"Feature registry: {len(self.features)} unique features, "
            f"{len(self.room_types)} room types (from PostgreSQL)"
        )

    def get_features_list(self) -> list[str]:
        return sorted(self.features)

    def get_room_types_list(self) -> list[str]:
        return sorted(self.room_types)

    def get_features_by_room(self, room_type: str) -> list[str]:
        return sorted(self.features_by_room_type.get(room_type, set()))


# Global singleton
registry = FeatureRegistry()
