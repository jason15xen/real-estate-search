from pydantic import BaseModel


class Address(BaseModel):
    Street: str
    District: str
    City: str
    State: str
    PostalCode: str
    Country: str
    Latitude: float
    Longitude: float


class RoomInstance(BaseModel):
    Features: list[str]


class Room(BaseModel):
    Type: str
    Count: int
    Instances: list[RoomInstance]


class Property(BaseModel):
    Name: str
    Address: Address
    AreaSqft: int
    PriceUSD: int
    Rooms: list[Room]

    def get_rooms_by_type(self, room_type: str) -> Room | None:
        for room in self.Rooms:
            if room.Type.lower() == room_type.lower():
                return room
        return None

    def get_room_count(self, room_type: str) -> int:
        room = self.get_rooms_by_type(room_type)
        return room.Count if room else 0

    def get_all_features(self) -> list[str]:
        features = []
        for room in self.Rooms:
            for instance in room.Instances:
                features.extend(instance.Features)
        return features

    def get_features_by_room_type(self, room_type: str) -> list[str]:
        room = self.get_rooms_by_type(room_type)
        if not room:
            return []
        features = []
        for instance in room.Instances:
            features.extend(instance.Features)
        return features

    def to_text_description(self) -> str:
        """Converts the property into a natural language description for embedding."""
        parts = [
            f"{self.Name} at {self.Address.Street}, {self.Address.City}, "
            f"{self.Address.State}, {self.Address.Country}.",
            f"Area: {self.AreaSqft} sqft. Price: ${self.PriceUSD:,}.",
        ]
        for room in self.Rooms:
            features_by_instance = []
            for i, instance in enumerate(room.Instances, 1):
                if instance.Features:
                    features_by_instance.append(
                        f"  Instance {i}: {', '.join(instance.Features)}"
                    )
            parts.append(
                f"{room.Count} {room.Type}(s):"
            )
            parts.extend(features_by_instance)
        return "\n".join(parts)
