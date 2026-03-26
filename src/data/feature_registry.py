"""
Feature Registry — Collects all unique feature names and room types from the data
at startup and stores them in memory.

This allows the query parser to map user input to exact feature names,
eliminating the need for vector search and LLM validation.
"""

import logging

from src.models.property import Property

logger = logging.getLogger(__name__)


class FeatureRegistry:
    def __init__(self):
        self.features: set[str] = set()
        self.room_types: set[str] = set()
        self.features_by_room_type: dict[str, set[str]] = {}

    def build_from_properties(self, properties: list[Property]) -> None:
        for prop in properties:
            for room in prop.Rooms:
                self.room_types.add(room.Type)
                if room.Type not in self.features_by_room_type:
                    self.features_by_room_type[room.Type] = set()
                for instance in room.Instances:
                    for feature in instance.Features:
                        self.features.add(feature)
                        self.features_by_room_type[room.Type].add(feature)

        logger.info(
            f"Feature registry: {len(self.features)} unique features, "
            f"{len(self.room_types)} room types"
        )

    def get_features_list(self) -> list[str]:
        return sorted(self.features)

    def get_room_types_list(self) -> list[str]:
        return sorted(self.room_types)

    def get_features_by_room(self, room_type: str) -> list[str]:
        return sorted(self.features_by_room_type.get(room_type, set()))


# Global singleton
registry = FeatureRegistry()
