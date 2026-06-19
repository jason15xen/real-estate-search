"""
Feature resolver — turns a user's raw feature phrase into the set of canonical
DB feature strings that genuinely represent it.

Two-stage retrieve-then-rerank:
  1. RECALL  — embed the phrase, pgvector top-K nearest features (broad net).
  2. PRECISION — one bundled LLM call filters each phrase's candidates down
     to the ones that truly mean what the user said (drops topical noise like
     "pool table" for "swimming pool").

The curated list per phrase is then used by the orchestrator's Phase 4 exactly
like the legacy registry alternatives were — same set-intersection (positive)
and set-subtraction (negated) logic, so the |with X| + |without X| = |total|
symmetry is preserved (positive and negated criteria with the same phrase get
the same curated list).
"""

from __future__ import annotations

import asyncio
import json
import logging

from config.settings import settings
from src.llm_client import embed_texts, get_query_client
from src.search.feature_index import retrieve_candidates

logger = logging.getLogger(__name__)

# Per-process cache: phrase (lowercased) -> curated feature list.
# Cleared whenever the feature embeddings change (see clear_cache()).
_cache: dict[str, list[str]] = {}
_CACHE_MAX = 5000  # bound memory; phrases are short, this is plenty


def clear_cache() -> None:
    """Drop the resolver cache — call when feature_embeddings is re-synced so a
    newly-embedded feature can start appearing in retrievals."""
    _cache.clear()


_RESOLVER_SYSTEM_PROMPT = """\
You are a real estate feature matcher. You are given several USER PHRASES, each \
with a list of CANDIDATE feature names retrieved from a property database.

For each phrase, select EVERY candidate that a listing could be tagged with that \
would satisfy a user searching for that phrase. Be INCLUSIVE: keep all genuine \
matches, including more-specific variants. Only drop candidates that are clearly \
a different thing, a wrong attribute, or merely topically-related noise. Missing \
a real match (under-matching) is worse than keeping a close one.

Decision rules (apply in order):
1. SUPERSET MATCH — if a candidate contains all the meaningful words of the \
phrase (e.g. phrase "covered pool" vs candidate "covered pool deck", "covered \
pool cage", "covered pool area"), KEEP it. These are specific kinds of the thing.
2. SYNONYM / SUBTYPE — keep candidates that are the same thing or a more specific \
kind, even with different words. e.g. phrase "covered pool" → also keep \
"enclosed pool", "screen-enclosed pool", "screened pool", "pool cage", \
"screened pool enclosure", "lanai pool" (a lanai is a covered structure). \
phrase "swimming pool" → keep "pool", "in-ground pool", "saltwater pool".
3. RESPECT ATTRIBUTES — a covered pool is a pool that is sheltered/roofed/ \
screened/enclosed. A "pool cover" or "pool tarp" is a separate accessory, NOT a \
covered pool — DROP those. DROP plain "pool" / "open-air pool" (no cover).
4. RESPECT COLORS / MATERIALS — phrase "yellow cabinets" → keep "yellow \
cabinets", "yellow kitchen cabinets", "mustard cabinets"; DROP "white cabinets", \
"shaker cabinets" (wrong color).
5. DROP TOPICAL NOISE — phrase "swimming pool" → DROP "pool table", "pool bath", \
"poolside shed".

Use ONLY strings that appear VERBATIM in that phrase's candidate list. Do not \
invent or alter strings. If genuinely none of the candidates fit, return an \
empty list for that phrase.

Return STRICT JSON, an object mapping each phrase to its filtered array:
{"phrase one": ["feat", ...], "phrase two": [...]}
No commentary, no markdown."""


async def _retrieve_for_phrases(pool, phrases: list[str], top_k: int) -> dict[str, list[str]]:
    """Embed each phrase and pgvector-retrieve top_k candidates. Returns
    {phrase: [candidate features]}. Phrases are embedded in one batched call."""
    vectors = await embed_texts(phrases)
    out: dict[str, list[str]] = {}

    async def _one(phrase: str, vec: list[float]):
        async with pool.acquire() as conn:
            out[phrase] = await retrieve_candidates(conn, vec, top_k)

    await asyncio.gather(*[_one(p, v) for p, v in zip(phrases, vectors)])
    return out


async def _llm_filter(candidates_by_phrase: dict[str, list[str]]) -> dict[str, list[str]]:
    """Bundled LLM #2 call (OpenAI): filter each phrase's candidates to
    the relevant subset. Falls back to returning the raw candidates on any failure
    (recall over precision when the filter is unavailable)."""
    # Only send phrases that actually retrieved candidates.
    payload = {p: c for p, c in candidates_by_phrase.items() if c}
    if not payload:
        return {p: [] for p in candidates_by_phrase}

    client = get_query_client()
    user_msg = json.dumps(payload, ensure_ascii=False)
    try:
        resp = await client.chat.completions.create(
            model=settings.openai_model_for_query,
            # Generous ceiling: the output echoes a filtered subset of up to
            # (num_phrases * top_k) feature strings. Billing is per token actually
            # generated, so a high ceiling costs nothing on typical small outputs.
            max_completion_tokens=16384,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _RESOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        filtered = json.loads(text)
    except Exception as e:
        logger.warning("LLM #2 feature filter failed (%s); using raw candidates", e)
        return dict(candidates_by_phrase)

    # Sanitize: keep only strings that were genuinely in each phrase's candidate
    # list (guards against the model inventing or altering strings).
    result: dict[str, list[str]] = {}
    for phrase, cands in candidates_by_phrase.items():
        cand_set = set(cands)
        picked = filtered.get(phrase)
        if isinstance(picked, list):
            # Respect the model's filtering — even an empty result. The
            # orchestrator falls back to an exact-match on the raw phrase when
            # the curated list is empty, which is the precision-preserving
            # choice (flooding all 200 candidates would reintroduce the noise
            # the LLM #2 filter exists to remove).
            result[phrase] = [f for f in picked if f in cand_set]
        else:
            # Phrase missing from the response entirely → treat as a filter
            # failure for this phrase and fall back to raw candidates (recall).
            result[phrase] = cands
    return result


async def resolve_feature_phrases(
    pool,
    phrases: list[str],
    top_k: int | None = None,
) -> dict[str, list[str]]:
    """Map each distinct user feature phrase to its curated DB-feature list.

    Returns {phrase: [canonical feature strings]}. Cached per phrase; cache is
    cleared when feature embeddings re-sync.
    """
    if not phrases:
        return {}
    k = top_k or settings.search_embedding_top_k

    # De-dupe by lowercased phrase, honor cache.
    distinct: dict[str, str] = {}        # lowercased -> original phrase
    for p in phrases:
        key = p.strip().lower()
        if key and key not in distinct:
            distinct[key] = p

    to_resolve = [orig for key, orig in distinct.items() if key not in _cache]

    if to_resolve:
        retrieved = await _retrieve_for_phrases(pool, to_resolve, k)
        filtered = await _llm_filter(retrieved)
        if len(_cache) > _CACHE_MAX:   # simple bound — drop all and refill
            _cache.clear()
        for phrase in to_resolve:
            _cache[phrase.strip().lower()] = filtered.get(phrase, [])

    # Assemble result keyed by the ORIGINAL phrase strings the caller passed.
    return {p: _cache.get(p.strip().lower(), []) for p in phrases}
