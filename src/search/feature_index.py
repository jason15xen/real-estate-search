"""
Feature embedding index — keeps `feature_embeddings` in sync with the distinct
feature strings present in room_instances, and provides cosine-similarity
retrieval over them.

Design notes:
  * Vectors are passed to/from Postgres as string literals with a ::vector cast
    (e.g. '[0.1,0.2,...]'). This avoids adding the `pgvector` Python dependency
    and an asyncpg type codec — we never need to DECODE a vector in Python,
    only retrieve the `feature` text from a similarity search.
  * sync_feature_embeddings is incremental and idempotent: it embeds only
    features that aren't already in feature_embeddings. Safe to call at startup
    and on every NOTIFY feature_change.
"""

from __future__ import annotations

import logging

from src.llm_client import embed_texts

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 256  # inputs per Azure embedding API call


def vector_literal(vec: list[float]) -> str:
    """Format a float list as a pgvector string literal: '[v1,v2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


async def _distinct_db_features(conn) -> set[str]:
    """All distinct feature strings currently in room_instances (excludes the
    empty-feature 'Unknown' stubs naturally — they contribute no features)."""
    rows = await conn.fetch(
        "SELECT DISTINCT unnest(features) AS feature FROM room_instances"
    )
    return {r["feature"] for r in rows if r["feature"]}


async def _embedded_features(conn) -> set[str]:
    rows = await conn.fetch("SELECT feature FROM feature_embeddings")
    return {r["feature"] for r in rows}


async def sync_feature_embeddings(pool) -> int:
    """Embed any DB features not yet present in feature_embeddings.

    Incremental + idempotent. Returns the number of newly-embedded features.
    """
    async with pool.acquire() as conn:
        db_features = await _distinct_db_features(conn)
        have = await _embedded_features(conn)

    missing = sorted(db_features - have)
    if not missing:
        logger.info("Feature embeddings up to date (%d features)", len(db_features))
        return 0

    logger.info("Embedding %d new features (of %d total)...", len(missing), len(db_features))
    embedded = 0
    for start in range(0, len(missing), EMBED_BATCH_SIZE):
        batch = missing[start:start + EMBED_BATCH_SIZE]
        try:
            vectors = await embed_texts(batch)
        except Exception as e:
            logger.error("Embedding batch failed (%d items): %s", len(batch), e)
            continue
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO feature_embeddings (feature, embedding)
                VALUES ($1, $2::vector)
                ON CONFLICT (feature) DO NOTHING
                """,
                [(feat, vector_literal(vec)) for feat, vec in zip(batch, vectors)],
            )
        embedded += len(batch)
        logger.info("  embedded %d/%d", min(start + EMBED_BATCH_SIZE, len(missing)), len(missing))

    logger.info("Feature-embedding sync complete: %d newly embedded", embedded)
    return embedded


async def retrieve_candidates(conn, query_vec: list[float], k: int) -> list[str]:
    """Return the top-k feature strings most cosine-similar to query_vec.

    Uses pgvector's <=> cosine-distance operator. At small scale the planner
    does an exact seq-scan sort; once the table is large enough to use the
    HNSW index, `hnsw.ef_search` bounds how many candidates the index
    considers — it MUST be >= k or the top-k silently degrades. We set it per
    query inside a transaction (SET LOCAL); it's a harmless no-op while the
    planner is still seq-scanning.
    """
    ef_search = int(max(k, 100))
    async with conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
        rows = await conn.fetch(
            """
            SELECT feature
            FROM feature_embeddings
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vector_literal(query_vec),
            k,
        )
    return [r["feature"] for r in rows]
