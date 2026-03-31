"""
Pydantic models for the image analyzer endpoint.

Request format matches data.json — raw Zillow property records.
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
    features: list[str]


class PropertyResult(BaseModel):
    id: str
    total_photos: int
    processed_photos: int
    rooms: list[PhotoResult]


class ProcessResponse(BaseModel):
    total_properties: int
    results: list[PropertyResult]


class SaveResponse(BaseModel):
    total_properties: int
    total_rooms: int
    total_room_instances: int
    total_schools: int
    message: str
