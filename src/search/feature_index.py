"""Feature embedding index: keeps `feature_embeddings` in sync with distinct room_instances features and provides cosine-similarity retrieval (vectors passed as ::vector string literals)."""

from __future__ import annotations

import logging

from src.llm_client import embed_texts

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 256  # inputs per embedding API call


def vector_literal(vec: list[float]) -> str:
    """Format a float list as a pgvector string literal: '[v1,v2,...]'."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


async def _distinct_db_features(conn) -> set[str]:
    """All distinct feature strings currently in room_instances."""
    rows = await conn.fetch(
        "SELECT DISTINCT unnest(features) AS feature FROM room_instances"
    )
    return {r["feature"] for r in rows if r["feature"]}


async def _embedded_features(conn) -> set[str]:
    rows = await conn.fetch("SELECT feature FROM feature_embeddings")
    return {r["feature"] for r in rows}


async def sync_feature_embeddings(pool) -> int:
    """Embed any DB features not yet in feature_embeddings (incremental + idempotent); returns the count newly embedded."""
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
    """Return the top-k feature strings most cosine-similar to query_vec (sets hnsw.ef_search >= k per query so the HNSW index doesn't silently degrade top-k)."""
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
