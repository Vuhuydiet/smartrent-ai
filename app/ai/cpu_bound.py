"""
Process-wide limiter for CPU-bound work.

The service runs as a single uvicorn worker, so CPU-bound calls (TF-IDF
vectorization, perceptual-hash image decode) executed directly inside async
handlers block the event loop and, under a burst, pin a core with no throughput
gain (the GIL serializes them anyway).

`run_cpu_bound` offloads such a call to a worker thread (numpy/sklearn/Pillow
release the GIL during compute, so the event loop stays responsive) and bounds
how many run at once via a shared semaphore. This caps peak CPU regardless of how
many requests arrive concurrently, and protects the other endpoints that share
the worker (chat, price suggestion).

Tune with the CPU_BOUND_CONCURRENCY env var (default 2).
"""

import asyncio
from typing import Any, Callable, TypeVar

from app.core.config import settings

_T = TypeVar("_T")

# Module-level semaphore — binds to the single long-lived uvicorn event loop on
# first use. Safe for a single-worker deployment.
_SEMAPHORE = asyncio.Semaphore(max(1, settings.CPU_BOUND_CONCURRENCY))


async def run_cpu_bound(fn: Callable[..., _T], *args: Any) -> _T:
    """Run a CPU-bound sync function off the event loop, bounded to N concurrent."""
    async with _SEMAPHORE:
        return await asyncio.to_thread(fn, *args)
