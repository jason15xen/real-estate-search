"""
Pydantic models for the image-analyzer ingest flow.

The single endpoint is POST /process; it accepts a list of `PropertyInput`
({id, data}). `PropertyItem` is the internal shape passed to the analyzer
and primary-table writers.
"""

from pydantic import BaseModel


class PhotoSource(BaseModel):
    url: str
    width: int


class MixedSources(BaseModel):
    jpeg: list[PhotoSource] = []
    webp: list[PhotoSource] = []


class Photo(BaseModel):
    caption: str = ""
    mixedSources: MixedSources


class ZillowPropertyRecord(BaseModel):
    model_config = {"extra": "allow"}

    originalPhotos: list[Photo] = []


class PropertyItem(BaseModel):
    model_config = {"extra": "allow"}

    Id: str
    ZillowPropertyId: int = 0
    ZillowPropertyRecord: ZillowPropertyRecord


class PhotoResult(BaseModel):
    photo_url: str
    room_type: str
    color: str | None = None  # one of 13 palette colors, or None / "Unknown"
    features: list[str]


class PropertyInput(BaseModel):
    """Request item shape for POST /process.

    - `id`: property's database identifier (GUID)
    - `data`: the full Zillow property record (address, price, originalPhotos,
      schools, resoFacts, etc.)
    """
    id: str
    data: dict

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "abc-123-def-456",
                "data": {
                    "address": {
                        "streetAddress": "123 Main St",
                        "city": "Titusville",
                        "state": "FL",
                        "zipcode": "32796",
                        "subdivision": "Sample Subdivision",
                    },
                    "latitude": 28.6,
                    "longitude": -80.8,
                    "price": 500000,
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "livingArea": 1800,
                    "homeType": "SINGLE_FAMILY",
                    "yearBuilt": 1995,
                    "description": "...",
                    "originalPhotos": [
                        {"mixedSources": {"jpeg": [{"url": "https://...", "width": 1536}]}}
                    ],
                    "schools": [
                        {"name": "Example School", "rating": 8, "grades": "K-5", "distance": 0.6, "link": "..."}
                    ],
                    "resoFacts": {
                        "stories": 1,
                        "hasPrivatePool": True,
                        "hasWaterfrontView": False,
                        "listingTerms": "Cash,Conventional,FHA"
                    },
                },
            }
        }
    }
