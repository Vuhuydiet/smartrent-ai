"""
Perceptual-hash (pHash) image similarity for duplicate listing detection.

Offline, best-effort image comparison used as a booster signal in Step 3 of the
duplicate detection pipeline. For the ≤ N suspicious candidates already fetched
for LLM confirmation, we download a bounded number of images per listing, compute
a perceptual hash, and take the best pairwise match against the new listing.

Every operation degrades gracefully: a failed download, a non-image payload, an
oversized file, or a missing library all reduce to a neutral similarity of 0.0
and never raise — the pipeline must never be blocked by image handling.
"""

import asyncio
import io
import logging
from typing import Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Bounds — keep per-check image work small (Step 3 runs on ≤ _MAX_LLM_CHECKS
# candidates, each downloading up to _IMG_MAX images).
_IMG_MAX = 5
_IMG_TIMEOUT = 5.0
_IMG_MAX_BYTES = 5_000_000
_PHASH_BITS = 64  # imagehash.phash default hash_size=8 → 8*8 = 64 bits


def _phash(image_bytes: bytes) -> Optional[Any]:
    """Compute a perceptual hash from raw image bytes, or None on any failure."""
    try:
        import imagehash  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        with Image.open(io.BytesIO(image_bytes)) as img:
            return imagehash.phash(img)
    except Exception as e:  # noqa: BLE001 — best-effort, never propagate
        logger.debug("pHash computation failed: %s", e)
        return None


async def fetch_and_hash(url: str, client: httpx.AsyncClient) -> Optional[Any]:
    """Download an image and return its perceptual hash, or None on any failure."""
    if not url:
        return None
    try:
        response = await client.get(url, timeout=_IMG_TIMEOUT)
        response.raise_for_status()
        content = response.content
        if not content or len(content) > _IMG_MAX_BYTES:
            return None
        return _phash(content)
    except Exception as e:  # noqa: BLE001 — best-effort, never propagate
        logger.debug("Image fetch failed for %s: %s", str(url)[:120], e)
        return None


async def hash_image_set(urls: List[str], *, limit: int = _IMG_MAX) -> List[Any]:
    """
    Hash up to `limit` image URLs concurrently, dropping any that fail.

    Reuses a single AsyncClient so the whole set shares one connection pool.
    """
    selected = [u for u in (urls or []) if u][:limit]
    if not selected:
        return []

    async with httpx.AsyncClient(timeout=_IMG_TIMEOUT) as client:
        results = await asyncio.gather(*(fetch_and_hash(u, client) for u in selected))
    return [h for h in results if h is not None]


def best_image_similarity(a: List[Any], b: List[Any]) -> float:
    """
    Best pairwise perceptual-hash similarity between two hash sets (0.0-1.0).

    similarity = 1 - hammingDistance / 64. Returns 0.0 if either set is empty.
    """
    if not a or not b:
        return 0.0

    best = 0.0
    for h1 in a:
        for h2 in b:
            try:
                distance = h1 - h2  # imagehash overloads __sub__ → Hamming distance
            except Exception:  # noqa: BLE001 — mismatched hash sizes etc.
                continue
            sim = 1.0 - (distance / _PHASH_BITS)
            if sim > best:
                best = sim
    return round(best, 4)
