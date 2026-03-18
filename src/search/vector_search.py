"""
Vector Search — Uses ChromaDB to find properties with semantically similar features.

This handles the synonym/fuzzy matching problem:
  "covered pool" should match "enclosed pool" or "indoor pool"
  "hardwood floors" should match "wood flooring"

The vector search retrieves CANDIDATES. It does NOT make the final decision —
the LLM validator (validator.py) will verify each candidate actually matches.
"""

import logging

import chromadb

from config.settings import settings
from src.models.property import Property
from src.models.search import Criterion, FeatureCriterion

logger = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(
            name="property_features",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def search_by_features(
    criteria: list[Criterion],
    candidate_property_names: list[str] | None = None,
) -> list[str]:
    """
    Given feature criteria, find property names whose features are
    semantically similar. Returns property names ranked by relevance.

    If candidate_property_names is provided, only searches within those
    properties (used after hard filters narrow the set).
    """
    feature_criteria = [c for c in criteria if isinstance(c, FeatureCriterion)]
    if not feature_criteria:
        return candidate_property_names or []

    collection = get_collection()

    # Build a combined query from all feature criteria
    query_parts = []
    for fc in feature_criteria:
        if fc.room_context:
            query_parts.append(f"{fc.room_context} with {fc.feature}")
        else:
            query_parts.append(fc.feature)

    query_text = "; ".join(query_parts)

    # Build where filter to restrict to candidates if provided
    where_filter = None
    if candidate_property_names:
        if len(candidate_property_names) == 1:
            where_filter = {"property_name": candidate_property_names[0]}
        else:
            where_filter = {"property_name": {"$in": candidate_property_names}}

    results = collection.query(
        query_texts=[query_text],
        n_results=min(settings.vector_search_top_k, collection.count()),
        where=where_filter if where_filter else None,
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    # Extract unique property names in order, preserving rank
    seen = set()
    property_names = []
    for metadata in results["metadatas"][0]:
        name = metadata["property_name"]
        if name not in seen:
            seen.add(name)
            property_names.append(name)

    logger.info(
        f"Vector search for '{query_text}': found {len(property_names)} candidate properties"
    )
    return property_names
