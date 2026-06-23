"""Feature Registry — caches unique feature names and room types from PostgreSQL for mapping user input to exact names."""

import logging

import asyncpg

logger = logging.getLogger(__name__)


class FeatureRegistry:
    def __init__(self):
        self.features: set[str] = set()
        self.room_types: set[str] = set()
        self.features_by_room_type: dict[str, set[str]] = {}
        # Pure-function cache; cleared on every build_from_db().
        self._alternatives_cache: dict[str, list[str]] = {}

    async def build_from_db(self, pool: asyncpg.Pool) -> None:
        """Load unique features and room types from PostgreSQL, building locally then atomic-swapping in to avoid torn reads."""
        new_features: set[str] = set()
        new_room_types: set[str] = set()
        new_by_room: dict[str, set[str]] = {}

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT room_type FROM room_instances ORDER BY room_type"
            )
            for row in rows:
                new_room_types.add(row["room_type"])

            rows = await conn.fetch("""
                SELECT DISTINCT room_type, unnest(features) AS feature
                FROM room_instances
                ORDER BY room_type, feature
            """)
            for row in rows:
                feature = row["feature"]
                room_type = row["room_type"]
                new_features.add(feature)
                new_by_room.setdefault(room_type, set()).add(feature)

        # Atomic swap — no torn read.
        self.features = new_features
        self.room_types = new_room_types
        self.features_by_room_type = new_by_room
        self._alternatives_cache = {}

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

    def get_feature_alternatives(self, base: str) -> list[str]:
        """Return all known features meaning "property HAS `base`" — pure, cached, deterministic; matches every word of `base` while excluding tangential forms."""
        if base in self._alternatives_cache:
            return self._alternatives_cache[base]

        base_lower = base.lower().strip()
        if not base_lower:
            self._alternatives_cache[base] = []
            return []
        base_words = set(base_lower.split())
        if not base_words:
            self._alternatives_cache[base] = []
            return []

        EXCLUSION_SUBSTRINGS = {
            "view", "viewing",
            "table", "tables",
            "community", "communities",
            "nearby", "neighbor", "neighboring", "neighborhood",
            "room for", "space for", "ready for",
            "potential",
        }

        # Skip exclusions present in the query itself (e.g. "community pool").
        effective_exclusions = {ex for ex in EXCLUSION_SUBSTRINGS if ex not in base_lower}

        result: list[str] = [base] if base not in self.features else []
        for feat in self.features:
            feat_lower = feat.lower()
            feat_words = set(feat_lower.split())
            if not base_words.issubset(feat_words):
                continue
            if any(ex in feat_lower for ex in effective_exclusions):
                continue
            result.append(feat)

        seen = set()
        unique = []
        for f in result:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        sorted_result = sorted(unique)
        self._alternatives_cache[base] = sorted_result
        return sorted_result


registry = FeatureRegistry()
