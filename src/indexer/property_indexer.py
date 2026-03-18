"""
Property Indexer — Loads property data into ChromaDB for vector search.

Each property's rooms and features are indexed as separate documents,
enabling feature-level semantic search. The property_name metadata
links each document back to its parent property.
"""

import logging

from src.models.property import Property
from src.search.vector_search import get_collection

logger = logging.getLogger(__name__)


def index_properties(properties: list[Property]) -> int:
    """
    Indexes all properties into ChromaDB. Each room instance becomes a
    separate document for fine-grained feature matching.

    Returns the total number of documents indexed.
    """
    collection = get_collection()

    # Clear existing data
    if collection.count() > 0:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    ids = []
    documents = []
    metadatas = []

    for prop in properties:
        # Index each room instance as a separate document
        for room in prop.Rooms:
            for i, instance in enumerate(room.Instances):
                doc_id = f"{prop.Name}_{room.Type}_{i}"
                feature_text = ", ".join(instance.Features)
                doc_text = (
                    f"{room.Type} in {prop.Name}: {feature_text}. "
                    f"Located at {prop.Address.City}, {prop.Address.State}."
                )

                ids.append(doc_id)
                documents.append(doc_text)
                metadatas.append({
                    "property_name": prop.Name,
                    "room_type": room.Type,
                    "instance_index": i,
                    "features": feature_text,
                    "city": prop.Address.City,
                    "state": prop.Address.State,
                })

    # Batch insert (ChromaDB handles embedding via its default model)
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    total = len(ids)
    logger.info(f"Indexed {total} documents from {len(properties)} properties")
    return total
