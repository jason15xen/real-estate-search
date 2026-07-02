"""OpenAI Batch API vision path: bulk photo analysis at 50% cost for large backlogs.

Flow: when many 'unprocessed' rows are pending, the worker packs their photos into one
JSONL (one chat.completions request per photo, custom_id "{item_id}|{photo_idx}"),
uploads it, creates a batch, marks the rows 'batch_submitted', and records the batch in
vision_batches. Each poll cycle it checks open batches; a completed (or expired-with-
partial-output) batch is parsed back into {photo_url: PhotoResult} per property and the
worker runs the normal full-process path with those results injected — photos missing
from the output (error-file lines) are re-analyzed synchronously inside that path.
Failed/cancelled batches, submit errors, and orphaned rows all reset to 'unprocessed'
so the sync path picks them up; nothing can get stuck in 'batch_submitted'.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import asyncpg

from config.settings import settings
from src.img_analyzer.analyzer import build_vision_system_prompt, parse_vision_content
from src.img_analyzer.models import PhotoResult
from src.img_analyzer.raw_db import extract_photo_urls_from_data
from src.llm_client import get_async_client

logger = logging.getLogger(__name__)

# Stay well under the Batch API's 200MB-per-input-file limit (the ~17KB system prompt
# repeats on every line, so ~8k photos ≈ 150MB).
MAX_BATCH_BYTES = 120 * 1024 * 1024

# Batches are also capped by ESTIMATED enqueued tokens (settings.vision_batch_max_tokens):
# OpenAI orgs have a batch queue token limit (observed live: 900k for this org — exceeding
# it fails the batch outright with token_limit_exceeded). Estimate per line =
# ~chars/4 for the prompt + image + completion ceiling.
_EST_TOKENS_IMAGE_AND_COMPLETION = 2_500  # high-detail image ~1.5k + 1k completion ceiling

# Batch statuses still moving at OpenAI; anything else is terminal.
_OPEN_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}

# The Batch API rejects custom_ids longer than 64 chars — and one bad line fails the
# WHOLE batch, so over-long item ids are routed to the sync path instead.
_CUSTOM_ID_MAX = 64

# After a submit error or a 'failed' batch, hold off new submissions so a deterministic
# failure (quota, invalid model, poison row) can't loop submit→fail→requeue→resubmit;
# the sync path keeps draining rows during the cooldown. Per-process, in-memory.
FAILURE_COOLDOWN_SECONDS = 600.0
_backoff_until_monotonic = 0.0

# Poll throttle (per process) so the 5s worker loop doesn't hammer the batches endpoint.
_last_poll_monotonic = 0.0

# One-time DDL guard per process (worker loops every ~5s).
_tables_ready = False


async def _ensure_tables(conn: asyncpg.Connection) -> None:
    """Create/migrate the vision_batches tracker and widen the raw_properties status CHECK."""
    global _tables_ready
    if _tables_ready:
        return
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS vision_batches (
            batch_id     TEXT PRIMARY KEY,
            status       TEXT NOT NULL DEFAULT 'submitted',
            items        JSONB NOT NULL,
            submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at TIMESTAMPTZ
        );
    """)
    await conn.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'raw_properties_status_check'
                  AND pg_get_constraintdef(oid) NOT ILIKE '%batch_submitted%'
            ) THEN
                ALTER TABLE raw_properties DROP CONSTRAINT raw_properties_status_check;
                ALTER TABLE raw_properties ADD CONSTRAINT raw_properties_status_check
                    CHECK (status IN (
                        'unprocessed',
                        'image_only_processed',
                        'partial_image_only_processed',
                        'processed',
                        'batch_submitted'
                    ));
            END IF;
        END $$;
    """)
    _tables_ready = True


def _request_line(item_id: str, idx: int, url: str, system_prompt: str) -> str:
    """One JSONL line: the exact same request body the sync path sends for this photo."""
    return json.dumps({
        "custom_id": f"{item_id}|{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": settings.openai_model,
            "max_completion_tokens": 1000,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                    ],
                },
            ],
        },
    })


async def submit(pool: asyncpg.Pool, rows: list[asyncpg.Record]) -> int:
    """Pack 'unprocessed' rows into one batch and submit it; marks each included row
    'batch_submitted' (guarded by updated_at so a concurrent /process write wins).
    Returns the number of properties actually batched; 0 = nothing submitted."""
    global _backoff_until_monotonic
    if time.monotonic() < _backoff_until_monotonic:
        return 0  # cooling down after a failed batch/submit; sync path drains meanwhile

    system_prompt = build_vision_system_prompt()
    est_tokens_per_line = len(system_prompt) // 4 + _EST_TOKENS_IMAGE_AND_COMPLETION

    # Build lines per item up to the byte AND estimated-token caps; photo-less rows and
    # rows whose id would overflow the custom_id limit stay on the sync path.
    per_item_lines: dict[str, list[str]] = {}
    per_item_urls: dict[str, list[str]] = {}
    total_bytes = 0
    total_est_tokens = 0
    for row in rows:
        data = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
        urls = extract_photo_urls_from_data(data)
        if not urls:
            continue
        if len(f"{row['id']}|{len(urls) - 1}") > _CUSTOM_ID_MAX:
            logger.warning("Batch: id %r too long for a custom_id — leaving to the sync path", row["id"])
            continue
        lines = [_request_line(row["id"], i, u, system_prompt) for i, u in enumerate(urls)]
        size = sum(len(l) + 1 for l in lines)
        est = len(lines) * est_tokens_per_line
        if per_item_lines and (total_bytes + size > MAX_BATCH_BYTES
                               or total_est_tokens + est > settings.vision_batch_max_tokens):
            break  # leftover rows stay pending; a later iteration batches them
        total_bytes += size
        total_est_tokens += est
        per_item_lines[row["id"]] = lines
        per_item_urls[row["id"]] = urls

    if not per_item_lines:
        return 0

    # Mark rows first (updated_at-guarded) so a row a concurrent write just changed is
    # dropped from this batch instead of having stale results applied later.
    items: dict[str, dict] = {}
    async with pool.acquire() as conn:
        await _ensure_tables(conn)
        for row in rows:
            if row["id"] not in per_item_lines:
                continue
            new_updated = await conn.fetchval(
                """
                UPDATE raw_properties SET status = 'batch_submitted', updated_at = NOW()
                WHERE id = $1 AND status = 'unprocessed' AND updated_at = $2
                RETURNING updated_at
                """,
                row["id"], row["updated_at"],
            )
            if new_updated is None:
                per_item_lines.pop(row["id"])
                continue
            items[row["id"]] = {
                "updated_at": new_updated.isoformat(),
                "urls": per_item_urls[row["id"]],
            }

    if not items:
        return 0

    jsonl = "\n".join(l for lines in per_item_lines.values() for l in lines)
    client = get_async_client()
    f = None
    try:
        f = await client.files.create(
            file=("vision_batch.jsonl", jsonl.encode("utf-8")), purpose="batch"
        )
        batch = await client.batches.create(
            input_file_id=f.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    except Exception as e:  # noqa: BLE001 — reset rows so the sync path takes over
        _backoff_until_monotonic = time.monotonic() + FAILURE_COOLDOWN_SECONDS
        logger.error("Batch submit failed (%s); returning %d rows to sync path and backing off %.0fs",
                     e, len(items), FAILURE_COOLDOWN_SECONDS)
        if f is not None:
            try:
                await client.files.delete(f.id)  # don't strand the uploaded JSONL
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE raw_properties SET status = 'unprocessed', updated_at = NOW()
                WHERE id = ANY($1) AND status = 'batch_submitted'
                """,
                list(items),
            )
        return 0

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO vision_batches (batch_id, status, items) VALUES ($1, $2, $3::jsonb)",
            batch.id, batch.status, json.dumps(items),
        )
    n_photos = sum(len(v["urls"]) for v in items.values())
    logger.info("Batch %s submitted: %d properties, %d photos (%.1f MB)",
                batch.id, len(items), n_photos, total_bytes / 1e6)
    return len(items)


async def has_open_batch(pool: asyncpg.Pool) -> bool:
    """True if a batch is still in flight. Submissions are serialized on this: org
    batch queues have an enqueued-token cap (observed 900k), so a second concurrent
    batch would be rejected with token_limit_exceeded anyway."""
    async with pool.acquire() as conn:
        await _ensure_tables(conn)
        return bool(await conn.fetchval(
            "SELECT 1 FROM vision_batches WHERE status IN "
            "('submitted', 'validating', 'in_progress', 'finalizing', 'cancelling') LIMIT 1"
        ))


async def recover_when_disabled(pool: asyncpg.Pool) -> int:
    """Requeue rows stranded in 'batch_submitted' by a previous run that had the batch
    flag on; call at worker startup when the flag is off (nothing else would ever
    claim them). Returns the number of rows requeued."""
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE raw_properties SET status = 'unprocessed', updated_at = NOW() "
            "WHERE status = 'batch_submitted'"
        )
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except ValueError:
        n = 0
    if n:
        logger.warning("Batch vision disabled: requeued %d stranded row(s) to the sync path", n)
    return n


async def _requeue_items(conn: asyncpg.Connection, item_ids: list[str]) -> None:
    """Return rows to the sync path (only ones still waiting on a batch)."""
    if item_ids:
        await conn.execute(
            """
            UPDATE raw_properties SET status = 'unprocessed', updated_at = NOW()
            WHERE id = ANY($1) AND status = 'batch_submitted'
            """,
            item_ids,
        )


async def _sweep_orphans(conn: asyncpg.Connection) -> None:
    """Requeue any 'batch_submitted' row no open batch is tracking (crash between a
    batch finishing and its results being applied). Runs before this cycle's batches
    are marked terminal, so rows about to be applied are never swept."""
    result = await conn.execute(
        """
        UPDATE raw_properties SET status = 'unprocessed', updated_at = NOW()
        WHERE status = 'batch_submitted'
          AND NOT EXISTS (
              SELECT 1 FROM vision_batches vb
              WHERE vb.status IN ('submitted', 'validating', 'in_progress', 'finalizing', 'cancelling')
                AND vb.items ? raw_properties.id
          )
        """
    )
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except ValueError:
        n = 0
    if n:
        logger.warning("Batch sweep: requeued %d orphaned row(s) to the sync path", n)


def _parse_output_jsonl(text: str, items: dict) -> dict[str, dict[str, PhotoResult]]:
    """Map output lines back to {item_id: {photo_url: PhotoResult}} via custom_id."""
    results: dict[str, dict[str, PhotoResult]] = {item_id: {} for item_id in items}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Per-line isolation: one malformed line (unexpected shape anywhere) must skip,
        # not raise — a raise here reads as a download failure and retries forever.
        try:
            rec = json.loads(line)
            item_id, _, idx_s = (rec.get("custom_id") or "").rpartition("|")
            info = items.get(item_id)
            if info is None:
                continue
            url = info["urls"][int(idx_s)]
            resp = rec.get("response") or {}
            if resp.get("status_code") != 200:
                continue  # left missing → sync re-analysis in the apply step
            body = resp.get("body") or {}
            choices = body.get("choices") or [{}]
            content = (choices[0].get("message") or {}).get("content") or ""
            results[item_id][url] = parse_vision_content(content, url)
        except (json.JSONDecodeError, ValueError, IndexError, KeyError, AttributeError, TypeError):
            continue
    return results


async def collect_ready(pool: asyncpg.Pool) -> list[dict]:
    """Poll open batches (throttled to vision_batch_poll_seconds); returns ready items as
    [{item_id, expected_updated_at, results}] for the worker to apply. Handles terminal
    states: completed/expired → parse whatever output exists; failed/cancelled → requeue."""
    global _last_poll_monotonic, _backoff_until_monotonic

    async with pool.acquire() as conn:
        await _ensure_tables(conn)
        open_rows = await conn.fetch(
            "SELECT batch_id, items FROM vision_batches "
            "WHERE status IN ('submitted', 'validating', 'in_progress', 'finalizing', 'cancelling')"
        )
        if not open_rows:
            await _sweep_orphans(conn)
            return []
        if time.monotonic() - _last_poll_monotonic < settings.vision_batch_poll_seconds:
            return []
        _last_poll_monotonic = time.monotonic()
        await _sweep_orphans(conn)

    client = get_async_client()
    ready: list[dict] = []
    for brow in open_rows:
        batch_id = brow["batch_id"]
        items = brow["items"] if isinstance(brow["items"], dict) else json.loads(brow["items"])
        try:
            batch = await client.batches.retrieve(batch_id)
        except Exception as e:  # noqa: BLE001 — transient unless the batch is gone
            if getattr(e, "status_code", None) == 404:
                # Batch no longer exists at OpenAI — it will never complete; requeue.
                logger.warning("Batch %s: not found at OpenAI — returning %d properties to the sync path",
                               batch_id, len(items))
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        await _requeue_items(conn, list(items))
                        await conn.execute(
                            "UPDATE vision_batches SET status = 'failed', completed_at = NOW() WHERE batch_id = $1",
                            batch_id,
                        )
            else:
                logger.warning("Batch %s: status check failed: %s", batch_id, e)
            continue

        if batch.status in _OPEN_STATUSES:
            counts = getattr(batch, "request_counts", None)
            logger.info("Batch %s: %s (%s/%s requests done)", batch_id, batch.status,
                        getattr(counts, "completed", "?"), getattr(counts, "total", "?"))
            continue

        if batch.status in ("failed", "cancelled"):
            if batch.status == "failed":
                # Likely deterministic (validation, quota, model) — pause new submissions
                # so requeued rows drain via sync instead of loop-resubmitting.
                _backoff_until_monotonic = time.monotonic() + FAILURE_COOLDOWN_SECONDS
            logger.warning("Batch %s: %s — returning %d properties to the sync path",
                           batch_id, batch.status, len(items))
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _requeue_items(conn, list(items))
                    await conn.execute(
                        "UPDATE vision_batches SET status = $2, completed_at = NOW() WHERE batch_id = $1",
                        batch_id, batch.status,
                    )
            continue

        # completed (or expired with partial output): use every result we got; the apply
        # step re-analyzes missing photos synchronously.
        results_by_item: dict[str, dict[str, PhotoResult]] = {i: {} for i in items}
        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            try:
                content = await client.files.content(output_file_id)
                results_by_item = _parse_output_jsonl(content.text, items)
            except Exception as e:  # noqa: BLE001 — transient unless the file is gone
                if getattr(e, "status_code", None) == 404:
                    # Output file deleted at OpenAI (retention/cleanup) — it will never
                    # download; requeue instead of retrying forever.
                    logger.warning("Batch %s: output file gone — returning %d properties to the sync path",
                                   batch_id, len(items))
                    async with pool.acquire() as conn:
                        async with conn.transaction():
                            await _requeue_items(conn, list(items))
                            await conn.execute(
                                "UPDATE vision_batches SET status = 'failed', completed_at = NOW() WHERE batch_id = $1",
                                batch_id,
                            )
                else:
                    logger.warning("Batch %s: output download failed: %s (will retry)", batch_id, e)
                continue
        elif batch.status == "expired":
            logger.warning("Batch %s: expired with no output — returning %d properties to sync",
                           batch_id, len(items))
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await _requeue_items(conn, list(items))
                    await conn.execute(
                        "UPDATE vision_batches SET status = 'expired', completed_at = NOW() WHERE batch_id = $1",
                        batch_id,
                    )
            continue

        # Atomic claim: only the process that flips the row open→terminal applies the
        # results, so two worker replicas can't double-apply the same batch.
        async with pool.acquire() as conn:
            claimed = await conn.fetchval(
                """
                UPDATE vision_batches SET status = $2, completed_at = NOW()
                WHERE batch_id = $1 AND status IN ('submitted', 'validating', 'in_progress', 'finalizing', 'cancelling')
                RETURNING batch_id
                """,
                batch_id, batch.status,
            )
        if claimed is None:
            continue  # another worker replica claimed this batch
        n_results = sum(len(v) for v in results_by_item.values())
        n_photos = sum(len(v["urls"]) for v in items.values())
        logger.info("Batch %s: %s — %d/%d photo results", batch_id, batch.status, n_results, n_photos)

        for item_id, info in items.items():
            ready.append({
                "item_id": item_id,
                "expected_updated_at": datetime.fromisoformat(info["updated_at"]),
                "results": results_by_item.get(item_id, {}),
            })
    return ready
