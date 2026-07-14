# Duplicate Detection — Perceptual-Hash Image Dedup + Recall Fixes

**Date:** 2026-07-14
**Repos:** `smartrent-ai` (primary), `smartrent-backend` (DTO field + notification logic)
**Status:** Design approved, pending implementation plan

## Background

The duplicate-listing detection pipeline runs inside AI auto-moderation
(`AiModerationProcessorServiceImpl.processSingleListing` → Feign
`POST /api/v1/listings/check-duplicate` → `DuplicateDetectionService`). A logic
review on 2026-07-14 found the wiring sound but detection **recall** weak, and
image-based duplicate detection absent. This spec covers two workstreams:

1. **Perceptual-hash (pHash) image dedup** — the strongest signal for
   photo-theft reposts, currently unimplemented (`imageUrls` accepted but ignored).
2. **Recall fixes** — three narrow gaps that cause true duplicates to be missed
   or under-surfaced.

Already applied in a prior pass (context, not part of this spec): price-band fix
(`minPrice`/`maxPrice` → `price="from..to"`), self-exclusion by `listingId`,
`fullNewAddress` fallback, removal of the misleading image line in the LLM prompt.

## Goals / Non-goals

**Goals:** add offline pHash image comparison as a booster signal; widen recall
via time window, product-type grouping, and symmetric admin notification.

**Non-goals:** unit tests / F1 measurement (deferred by request); precompute +
store hashes in DB (rejected — no migration/backfill for the DATN sprint);
multimodal-LLM image comparison; auto-reject of duplicates (flag-for-admin only).

## Design

### Component 1 — `app/ai/image_similarity.py` (new)

Mirrors `text_similarity.py`. Pure best-effort; every failure degrades to a
neutral value, never raises.

```
async def fetch_and_hash(url, client) -> ImageHash | None
    # httpx GET, timeout _IMG_TIMEOUT (5s), abort if > _IMG_MAX_BYTES (5MB),
    # Pillow open + imagehash.phash. Any error → None.

async def hash_image_set(urls, *, limit=_IMG_MAX) -> list[ImageHash]
    # hash up to `limit` urls concurrently (shared AsyncClient), drop Nones.

def best_image_similarity(a: list[ImageHash], b: list[ImageHash]) -> float
    # max over all pairs of (1 - hamming/64); 0.0 if either set empty.
```

Constants: `_IMG_MAX=5`, `_IMG_TIMEOUT=5.0`, `_IMG_MAX_BYTES=5_000_000`,
`_IMAGE_MATCH_MIN=0.85`, `_IMAGE_DRIVEN_CEILING=0.84` (strictly below
`_DUPLICATE_THRESHOLD`).

Dependencies: add `Pillow` + `imagehash` to project requirements. `numpy`/`scipy`
already present transitively via `scikit-learn`.

### Component 2 — integration in `DuplicateDetectionService._llm_confirm`

- Hash the new listing's images **once** from `new_listing["imageUrls"]`
  (bounded to `_IMG_MAX`), reusing a single `httpx.AsyncClient`.
- For each suspicious candidate (≤ `_MAX_LLM_CHECKS` = 3): the code already
  fetches `detail = await backend_client.get_listing(id)`. Extract candidate
  image URLs from `detail["media"]` (filter to image `mediaType`), hash them,
  compute `img_sim = best_image_similarity(new_hashes, cand_hashes)`.
- **Score combination (booster + floor):**

```
def _image_driven_score(img_sim: float) -> float:
    if img_sim < _IMAGE_MATCH_MIN:        # 0.85 — treat as different images
        return 0.0
    span = (img_sim - _IMAGE_MATCH_MIN) / (1.0 - _IMAGE_MATCH_MIN)
    # map [0.85, 1.0] → [SUSPICIOUS (0.6), CEILING (0.84)], strictly below DUPLICATE
    return round(_SUSPICIOUS_THRESHOLD + span * (_IMAGE_DRIVEN_CEILING - _SUSPICIOUS_THRESHOLD), 4)

match["score"] = max(match["score"], _image_driven_score(img_sim))
```

A near-identical photo guarantees the listing reaches admin review (≥ SUSPICIOUS,
capped at 0.84) but a single shared image alone **cannot reach** the DUPLICATE
line (0.85) without text/LLM agreement pushing `match["score"]` there —
bounding false positives from stock photos / logos. Only the real text/LLM score
can cross into DUPLICATE.

- Add `Mức giống nhau của ảnh: {pct}%` to the LLM prompt (a real computed signal,
  replacing the vague line previously removed). Requires computing `img_sim`
  before the LLM call.
- Add `imageSimilarity` to the emitted match (`_build_result`).

### Component 3 — recall fixes (`DuplicateDetectionService`)

- **Time window:** `postedWithinDays` 60 → constant `_CANDIDATE_WINDOW_DAYS = 180`.
- **Product-type grouping:** drop `productType` from the search params (single
  backend call) and filter candidates client-side to the same group:

```
_PRODUCT_TYPE_GROUPS = [{"ROOM", "STUDIO"}, {"APARTMENT", "OFFICE"}, {"HOUSE"}]
# a type in no group keeps its own hard match (group = {that type})
```

Price-band (now applied) + district + 180-day window already narrow the pool, so
one call suffices; clearly-different types (house vs room) are excluded so they
do not dilute scoring. Candidate `productType` is present on `ListingCardResponse`.

### Component 4 — symmetric admin notification (`AiModerationProcessorServiceImpl`)

```
flaggedDuplicate = duplicateResult != null
    && (DUPLICATE || SUSPICIOUS).equals(duplicateResult.getDecision())
```

- Send the admin duplicate notification whenever `flaggedDuplicate` and the
  verify outcome is not REJECTED (today it only fires when a would-be APPROVE was
  downgraded, so a NEEDS_REVIEW + DUPLICATE listing gets no duplicate-specific
  alert).
- Owner receives the "đang xem xét do nội dung tương tự" message when
  `flaggedDuplicate` and the listing is heading to review (downgraded APPROVE or
  already NEEDS_REVIEW).

### Component 5 — response DTOs

Add `imageSimilarity` (double, default 0) to:
- `DuplicateCheckResponse.SuspiciousMatch` (Java) —
  `smartrent-backend/.../dto/response/DuplicateCheckResponse.java`
- `SuspiciousMatch` (pydantic) — `smartrent-ai/app/api/v1/duplicate_check.py`
- `_build_result` in `duplicate_detection_service.py`

## Error handling

All image steps are best-effort: download failure, bad URL, non-image content,
oversized payload → that image is skipped, `img_sim` falls back toward 0.0, and
the pipeline proceeds on text/LLM signal alone. Consistent with the existing
invariant that duplicate detection never blocks moderation (fail = PASS).

## Verification

- `python -m py_compile` on all edited AI files.
- Gradle compile of `smartrent-backend` (DTO + notification changes).
- No automated tests / F1 in scope (deferred).

## Files touched

**smartrent-ai:** `app/ai/image_similarity.py` (new),
`app/service/duplicate_detection_service.py`, `app/api/v1/duplicate_check.py`,
requirements (`Pillow`, `imagehash`).

**smartrent-backend:** `dto/response/DuplicateCheckResponse.java`,
`service/ai/impl/AiModerationProcessorServiceImpl.java`.
