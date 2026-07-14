# AI Moderation — CPU Throttle for Large Backlogs

**Date:** 2026-07-14
**Repos:** `smartrent-backend` (scheduler batch size), `smartrent-ai` (CPU limiter)
**Status:** Design approved, pending implementation

## Problem

Draining a large one-time backlog (~20k listings from a seed/reshape) pins the
AI service CPU and makes the single shared VM sluggish.

Root cause: the moderation scheduler pulls a fixed batch of **20** listings every
5 minutes and processes them concurrently. Each listing fans out to `verify`
(Gemini multimodal — network I/O, offloaded) and `check-duplicate`. The duplicate
check runs **CPU-bound** work — `TfidfVectorizer.fit_transform` (char n-gram 2–4
over ~50 candidate descriptions) and, in Step 3, perceptual-hash image decode —
**synchronously inside the async handler** of a **single uvicorn worker** (the
Dockerfile runs `uvicorn` with no `--workers`). With 20 listings in flight, up to
20 CPU-bound tasks contend on one Python process. Because the GIL serializes
CPU-bound work, that concurrency does not increase throughput — it only thrashes
and pins a core, and it blocks the event loop so even the I/O-bound verify calls
stall.

## Constraints & goal

Single VM, fixed CPU (no added workers/replicas/cores). The backlog is one-time.
Priority: **reduce peak CPU** to keep the VM responsive; a slower drain is
acceptable. No re-architecture.

## Non-goals

Horizontal scaling / process pools (would consume more CPU, against the goal);
algorithm changes to TF-IDF or thresholds; disabling duplicate detection (a
separate operator lever, not in scope); load testing 20k here.

## Design

### 1. Backend — configurable scheduler batch size

`AiListingAutoModerationScheduler` currently hardcodes `PageRequest.of(0, 20)`.
Inject a configurable size (the delay is already configurable):

- Constructor `@Value("${smartrent.ai.verification.scheduler.batch-size:20}") int batchSize`.
- Use `PageRequest.of(0, batchSize)`.
- Declare the property in `application.yml` with **default 20** (current behavior
  preserved). For the drain, an operator lowers it to 3–5 — no code deploy. The
  existing admin enable/disable toggle allows controlled off-peak draining.

### 2. AI — shared CPU-bound limiter

New module `app/ai/cpu_bound.py` — a single process-wide limiter:

```python
import asyncio
from typing import Any, Callable, TypeVar
from app.core.config import settings

_T = TypeVar("_T")
_SEMAPHORE = asyncio.Semaphore(max(1, settings.CPU_BOUND_CONCURRENCY))

async def run_cpu_bound(fn: Callable[..., _T], *args: Any) -> _T:
    """Run a CPU-bound sync fn off the event loop, bounded to N concurrent."""
    async with _SEMAPHORE:
        return await asyncio.to_thread(fn, *args)
```

- `config.py`: add `CPU_BOUND_CONCURRENCY: int = 2` (env-tunable).
- `duplicate_detection_service.py`: run the TF-IDF hotspot through it —
  `scored = await run_cpu_bound(self._score_candidates, listing, candidates)`.
- `image_similarity.py`: run the pHash decode through it — inside `fetch_and_hash`,
  `return await run_cpu_bound(_phash, content)` (downloads stay concurrent; only
  the decode is bounded).

Effect: total CPU-bound concurrency across TF-IDF + image decode is capped at
`CPU_BOUND_CONCURRENCY` (default 2) regardless of how many requests arrive, so a
burst cannot pin the box. `to_thread` keeps the event loop free (numpy/sklearn/
Pillow release the GIL during compute), so the I/O-bound verify path is
unaffected and stays responsive. The limiter is process-wide, so it also protects
the chat / price-suggestion endpoints that share the worker.

## Error handling

`run_cpu_bound` changes only *where/when* a function runs, not its exception
semantics — exceptions propagate to the awaiter exactly as the direct call did.
The existing best-effort wrappers remain: `fetch_and_hash` still returns `None`
on failure; duplicate detection stays fail-safe (any failure → PASS, never blocks
moderation).

## Assumptions

Uvicorn runs a single long-lived event loop (true for the current deploy), so a
module-level `asyncio.Semaphore` binds to that loop safely.

## Verification

- `python -m py_compile` on edited/new AI files.
- `gradlew compileJava` on the backend.
- No automated tests (per request). Manual validation: lower `batch-size` and
  observe reduced peak CPU during a drain.

## Files touched

**smartrent-backend:** `cronjob/AiListingAutoModerationScheduler.java`,
`src/main/resources/application.yml`.

**smartrent-ai:** `app/ai/cpu_bound.py` (new), `app/core/config.py`,
`app/service/duplicate_detection_service.py`, `app/ai/image_similarity.py`.
