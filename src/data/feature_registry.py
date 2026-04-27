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
        # Pure-function cache: same registry state + same base → same result.
        # Cleared on every build_from_db() call.
        self._alternatives_cache: dict[str, list[str]] = {}

    async def build_from_db(self, pool: asyncpg.Pool) -> None:
        """Load unique features and room types from PostgreSQL."""
        # Reset registry state — old cached alternatives are no longer valid.
        self.features = set()
        self.room_types = set()
        self.features_by_room_type = {}
        self._alternatives_cache = {}

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

    def get_feature_alternatives(self, base: str) -> list[str]:
        """
        Deterministically compute all known features that mean "the property HAS `base`".

        Result is cached because the function is PURE — same registry state +
        same base always produces the same output. Cache is cleared whenever
        build_from_db() runs, so it can never go stale.

        - Starts from registry features that include every word of `base`.
        - Excludes tangential forms (views, game tables, accessories alone, \
neighborhood/neighbor references, "room for" placeholders).
        - Result is deterministic and identical for positive or negated queries, \
which guarantees: |with X| + |without X| = |total|.
        """
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

        result: list[str] = [base] if base not in self.features else []
        for feat in self.features:
            feat_lower = feat.lower()
            feat_words = set(feat_lower.split())
            if not base_words.issubset(feat_words):
                continue
            if any(ex in feat_lower for ex in EXCLUSION_SUBSTRINGS):
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


# Global singleton
registry = FeatureRegistry()
