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
    # Only use vector search for POSITIVE features — negated features skip this entirely
    feature_criteria = [
        c for c in criteria if isinstance(c, FeatureCriterion) and not c.negated
    ]
    if not feature_criteria:
        return candidate_property_names or []

    collection = get_collection()

    # Search each feature criterion separately to enforce room_type filtering
    per_criterion_results: list[set[str]] = []

    for fc in feature_criteria:
        query_text = (
            f"{fc.room_context} with {fc.feature}" if fc.room_context else fc.feature
        )

        # Build where filter: combine candidate restriction + room_type restriction
        conditions = []
        if candidate_property_names:
            if len(candidate_property_names) == 1:
                conditions.append({"property_name": candidate_property_names[0]})
            else:
                conditions.append({"property_name": {"$in": candidate_property_names}})
        if fc.room_context:
            conditions.append({"room_type": fc.room_context})

        where_filter = None
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        results = collection.query(
            query_texts=[query_text],
            n_results=min(settings.vector_search_top_k, collection.count()),
            where=where_filter if where_filter else None,
        )

        matched_names = set()
        if results["ids"] and results["ids"][0]:
            for metadata in results["metadatas"][0]:
                matched_names.add(metadata["property_name"])

        logger.info(
            f"Vector search for '{query_text}'"
            f"{f' (in {fc.room_context})' if fc.room_context else ''}"
            f": {len(matched_names)} matches"
        )
        per_criterion_results.append(matched_names)

    # Intersect: property must match ALL feature criteria
    if per_criterion_results:
        matched = per_criterion_results[0]
        for s in per_criterion_results[1:]:
            matched = matched & s
    else:
        matched = set()

    # Preserve order from first criterion's results
    property_names = [n for n in per_criterion_results[0] if n in matched] if matched else []

    logger.info(f"Vector search combined: {len(property_names)} candidate properties")
    return property_names
