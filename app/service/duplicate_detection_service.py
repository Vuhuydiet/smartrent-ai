"""
Duplicate Listing Detection Service — 3-step pipeline.

Step 1: Candidate retrieval — query backend for similar listings (same area, price range)
Step 2: Fast similarity scoring — TF-IDF + fuzzy match (Python, no LLM cost)
Step 3: LLM confirmation — Gemini reviews only suspicious matches (saves cost)

Decision thresholds:
  score >= 0.85  → DUPLICATE (flag for admin review; never auto-reject)
  score 0.6-0.85 → SUSPICIOUS (flag for admin review)
  score < 0.6    → PASS

Step 3 also folds in an offline perceptual-hash (pHash) image comparison as a
booster signal — see `_image_driven_score`.
"""

import logging
from typing import Any, Dict, List, Optional, Set, cast

from agents import Agent, Runner  # type: ignore[import]

from app.ai.cpu_bound import run_cpu_bound
from app.ai.image_similarity import best_image_similarity, hash_image_set
from app.ai.llm.agent_factory import default_model_settings, make_model
from app.ai.llm.gateway import get_gateway
from app.ai.text_similarity import batch_tfidf_similarity, compute_listing_similarity
from app.core import backend_client
from app.core.config import settings

logger = logging.getLogger(__name__)

_DUPLICATE_THRESHOLD = 0.85
_SUSPICIOUS_THRESHOLD = 0.6
_MAX_CANDIDATES = 50
_MAX_LLM_CHECKS = 3
_CANDIDATE_WINDOW_DAYS = 180

# Image-similarity thresholds (perceptual hash). A near-identical photo floors the
# score at SUSPICIOUS so the listing always reaches admin review, but capping the
# image-driven contribution strictly below _DUPLICATE_THRESHOLD ensures a single
# shared image (stock photo, logo) cannot reach DUPLICATE on its own — only the
# text/LLM score can cross that line.
_IMAGE_MATCH_MIN = 0.85
_IMAGE_DRIVEN_CEILING = 0.84

# Product types grouped by how easily one is mislabeled as another. Candidate
# retrieval keeps only listings whose productType shares a group with the new
# listing — catching cross-type reposts without diluting the pool with clearly
# different property kinds. A type in no group keeps its own exact match.
_PRODUCT_TYPE_GROUPS = [{"ROOM", "STUDIO"}, {"APARTMENT", "OFFICE"}, {"HOUSE"}]

_LLM_PROMPT = """\
Bạn là chuyên gia phát hiện tin đăng trùng lặp trên nền tảng bất động sản.

So sánh TIN MỚI với TIN CŨ bên dưới. Đánh giá xem chúng có phải là cùng một bất động sản không.

Tiêu chí đánh giá:
- Cùng địa chỉ hoặc vị trí rất gần nhau
- Mô tả giống nhau về nội dung (dù có thay đổi từ ngữ)
- Giá và diện tích tương tự
- Mức giống nhau của ảnh: {image_similarity_pct}% (100% = ảnh gần như trùng khớp)

TIN MỚI:
Tiêu đề: {new_title}
Mô tả: {new_desc}
Giá: {new_price} VND
Diện tích: {new_area} m²
Địa chỉ: {new_address}

TIN CŨ (ID: {old_id}):
Tiêu đề: {old_title}
Mô tả: {old_desc}
Giá: {old_price} VND
Diện tích: {old_area} m²
Địa chỉ: {old_address}

Trả lời CHÍNH XÁC bằng JSON:
{{"duplicate_score": <0.0-1.0>, "is_duplicate": <true/false>, "reason": "<giải thích ngắn>"}}
"""


class DuplicateDetectionService:
    """Orchestrates the 3-step duplicate detection pipeline."""

    def __init__(self) -> None:
        self._gateway = get_gateway()

    async def check_duplicate(
        self,
        listing: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check if a listing is a duplicate of any existing listing.

        Args:
            listing: Dict with keys: title, description, price, area,
                     productType, provinceCode, districtId, address,
                     imageUrls (optional)

        Returns:
            Dict with: isDuplicate, highestScore, decision, suspiciousMatches
        """
        logger.info(
            "Duplicate check for: '%s' in district=%s",
            listing.get("title", "")[:50],
            listing.get("districtId"),
        )

        # ── Step 1: Retrieve candidates from backend ──────────────────
        candidates = await self._retrieve_candidates(listing)

        # Defensive: never match the listing against itself. Today the public
        # search only returns moderationStatus=APPROVED and the listing under
        # moderation is not yet APPROVED, so it can't appear here — but a
        # re-moderation of an already-approved listing (or any future change to
        # the visibility gate) would otherwise self-match at score ~1.0.
        own_id = listing.get("listingId")
        if own_id is not None:
            candidates = [
                c for c in candidates if str(c.get("listingId")) != str(own_id)
            ]

        if not candidates:
            logger.info("No candidates found — listing is unique.")
            return _build_result("PASS", 0.0, [])

        logger.info("Step 1: %d candidates retrieved.", len(candidates))

        # ── Step 2: Fast similarity scoring ───────────────────────────
        # TF-IDF is CPU-bound; run it off the loop under the shared cap so a
        # batch burst can't pin the single worker's CPU.
        scored = await run_cpu_bound(self._score_candidates, listing, candidates)
        suspicious = [s for s in scored if s["score"] >= _SUSPICIOUS_THRESHOLD]

        if not suspicious:
            logger.info(
                "Step 2: No suspicious matches (highest=%.2f).",
                scored[0]["score"] if scored else 0.0,
            )
            highest = scored[0]["score"] if scored else 0.0
            return _build_result("PASS", highest, [])

        logger.info("Step 2: %d suspicious matches found.", len(suspicious))

        # ── Step 3: LLM confirmation for top suspicious ───────────────
        confirmed = await self._llm_confirm(listing, suspicious[:_MAX_LLM_CHECKS])

        highest_score = max(m["score"] for m in confirmed) if confirmed else 0.0
        decision = _decide(highest_score)

        logger.info(
            "Step 3: LLM confirmed %d matches. Decision=%s (score=%.2f)",
            len(confirmed),
            decision,
            highest_score,
        )

        return _build_result(decision, highest_score, confirmed)

    # ------------------------------------------------------------------
    # Step 1: Candidate retrieval
    # ------------------------------------------------------------------

    async def _retrieve_candidates(
        self, listing: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Query backend for listings similar in location and price.

        productType is intentionally NOT sent as a search filter — instead
        candidates are filtered client-side to the same product-type *group* so
        cross-type reposts (e.g. ROOM relisted as STUDIO) are still caught
        without diluting the pool with clearly different property kinds.
        """
        price = listing.get("price", 0)
        params: Dict[str, Any] = {
            "provinceCode": listing.get("provinceCode"),
            "size": _MAX_CANDIDATES,
            "postedWithinDays": _CANDIDATE_WINDOW_DAYS,
        }
        if listing.get("districtId"):
            params["districtId"] = listing["districtId"]
        if price:
            # ListingFilterRequest takes a single `price` range string "from..to"
            # (VND) — NOT minPrice/maxPrice, which it would silently drop, leaving
            # the ±20% band unapplied and diluting the 50-candidate pool.
            params["price"] = f"{int(price * 0.8)}..{int(price * 1.2)}"
        if listing.get("provinceCode"):
            params["provinceId"] = listing["provinceCode"]

        try:
            data = await backend_client.search_listings(params)
            if "error" in data:
                logger.warning("Backend search failed: %s", data["error"])
                return []
            listings = data.get("listings", [])
        except Exception as e:
            logger.error("Candidate retrieval failed: %s", e)
            return []

        group = _product_type_group(listing.get("productType"))
        if group:
            listings = [c for c in listings if c.get("productType") in group]
        return listings

    # ------------------------------------------------------------------
    # Step 2: Fast similarity scoring
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        new_listing: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score all candidates using TF-IDF + fuzzy matching."""
        new_desc = new_listing.get("description", "")
        candidate_descs = [c.get("description", "") for c in candidates]

        # Batch TF-IDF for descriptions (efficient)
        desc_scores = batch_tfidf_similarity(new_desc, candidate_descs)

        scored = []
        for i, candidate in enumerate(candidates):
            desc_tfidf = desc_scores[i] if i < len(desc_scores) else 0.0

            # Build comparable dicts
            new_data = {
                "title": new_listing.get("title", ""),
                "description": new_desc,
                "address": self._extract_address(new_listing),
                "price": new_listing.get("price", 0),
            }
            cand_data = {
                "title": candidate.get("title", ""),
                "description": candidate.get("description", ""),
                "address": self._extract_address(candidate),
                "price": candidate.get("price", 0),
            }

            combined_score, detail = compute_listing_similarity(
                new_data, cand_data, desc_tfidf_score=desc_tfidf
            )

            scored.append(
                {
                    "listingId": candidate.get("listingId"),
                    "title": candidate.get("title", ""),
                    "score": combined_score,
                    **detail,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    @staticmethod
    def _extract_address(listing: Dict[str, Any]) -> str:
        """Extract address string from listing (handles nested address object)."""
        addr = listing.get("address")
        if isinstance(addr, dict):
            return (
                addr.get("fullAddress", "")
                # New-structure cards populate fullNewAddress, not fullAddress.
                # Without this fallback, addr similarity is 0 for such candidates.
                or addr.get("fullNewAddress", "")
                or (
                    f'{addr.get("wardName", "")} '
                    f'{addr.get("districtName", "")} '
                    f'{addr.get("provinceName", "")}'
                ).strip()
            )
        return str(addr) if addr else ""

    # ------------------------------------------------------------------
    # Step 3: LLM confirmation
    # ------------------------------------------------------------------

    async def _llm_confirm(
        self,
        new_listing: Dict[str, Any],
        suspicious: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use Gemini + offline pHash to confirm or reject suspicious matches."""
        confirmed = []

        # Hash the new listing's images once, reused across all candidates.
        new_hashes = await hash_image_set(new_listing.get("imageUrls") or [])

        for match in suspicious:
            listing_id = match.get("listingId")
            match["imageSimilarity"] = 0.0
            img_sim = 0.0
            try:
                # Fetch full detail for the candidate
                detail = await backend_client.get_listing(str(listing_id))
                if "error" in detail:
                    match["llmScore"] = None
                    match["llmReason"] = "Could not fetch listing detail"
                    confirmed.append(match)
                    continue

                # Offline perceptual-hash image comparison (best-effort booster).
                cand_hashes = await hash_image_set(self._extract_image_urls(detail))
                img_sim = best_image_similarity(new_hashes, cand_hashes)
                match["imageSimilarity"] = img_sim

                prompt = _LLM_PROMPT.format(
                    new_title=new_listing.get("title", ""),
                    new_desc=(new_listing.get("description", ""))[:500],
                    new_price=new_listing.get("price", "N/A"),
                    new_area=new_listing.get("area", "N/A"),
                    new_address=self._extract_address(new_listing),
                    image_similarity_pct=round(img_sim * 100),
                    old_id=listing_id,
                    old_title=detail.get("title", ""),
                    old_desc=(detail.get("description", ""))[:500],
                    old_price=detail.get("price", "N/A"),
                    old_area=detail.get("area", "N/A"),
                    old_address=self._extract_address(detail),
                )

                trace = self._gateway.create_trace(
                    name="duplicate-check-llm",
                    metadata={"listingId": listing_id},
                )
                response_text = await self._run_llm_oneshot(
                    prompt, trace, span_name="duplicate-confirm"
                )

                llm_result = self._parse_llm_response(response_text)
                if llm_result:
                    match["llmScore"] = llm_result.get("duplicate_score")
                    match["llmReason"] = llm_result.get("reason", "")
                    # Update score with LLM weight
                    fast_score = match["score"]
                    llm_score = llm_result.get("duplicate_score", fast_score)
                    match["score"] = round(fast_score * 0.4 + llm_score * 0.6, 4)
                else:
                    match["llmScore"] = None
                    match["llmReason"] = "LLM response parse failed"

                # Image booster: a near-identical photo floors the score at
                # SUSPICIOUS (capped below DUPLICATE) so it always reaches admin.
                match["score"] = max(match["score"], _image_driven_score(img_sim))

            except Exception as e:
                logger.warning("LLM confirm failed for %s: %s", listing_id, e)
                match["llmScore"] = None
                match["llmReason"] = str(e)

            confirmed.append(match)

        confirmed.sort(key=lambda x: x["score"], reverse=True)
        return confirmed

    @staticmethod
    def _extract_image_urls(listing: Dict[str, Any]) -> List[str]:
        """Pull image URLs from a listing detail's `media` list (images only)."""
        urls: List[str] = []
        for m in listing.get("media") or []:
            if not isinstance(m, dict):
                continue
            # mediaType is "IMAGE" | "VIDEO"; keep images (and untyped, as a
            # lenient default) only.
            if m.get("mediaType") in (None, "IMAGE"):
                url = m.get("url")
                if url:
                    urls.append(url)
        return urls

    async def _run_llm_oneshot(self, prompt: str, trace: Any, *, span_name: str) -> str:
        """One-shot LLM call via the Agents SDK (no tools), returns raw text.

        Mirrors the pattern in GeminiListingVerificationHelper so the duplicate
        check uses the same provider-pluggable path (LiteLLM) and the same
        config key (LLM_CHAT_MODEL) as the rest of the post-migration service.
        """
        agent = Agent(
            name="Duplicate Checker",
            instructions=(
                "Bạn là chuyên gia phát hiện tin đăng bất động sản trùng lặp. "
                "Chỉ trả lời bằng JSON đúng định dạng được yêu cầu, "
                "không thêm bất kỳ chữ nào khác."
            ),
            model=make_model(settings.LLM_CHAT_MODEL),
            model_settings=default_model_settings(temperature=0.1),
        )

        generation = trace.generation(
            name=span_name,
            model=settings.LLM_CHAT_MODEL,
            input=prompt[:2000],
        )
        try:
            result = await Runner.run(
                starting_agent=agent,
                input=cast(Any, [{"role": "user", "content": prompt}]),
                max_turns=2,
            )
            text = str(result.final_output or "")
            generation.end(output=text[:2000])
            return text
        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            raise

    @staticmethod
    def _parse_llm_response(text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from the LLM's raw text output."""
        import json

        if not text:
            return None
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("\n```", 1)[0]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM duplicate response: %s", text[:200])
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decide(score: float) -> str:
    if score >= _DUPLICATE_THRESHOLD:
        return "DUPLICATE"
    if score >= _SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    return "PASS"


def _product_type_group(product_type: Optional[str]) -> Optional[Set[str]]:
    """Return the product-type group a type belongs to (for candidate filtering).

    Falls back to a singleton {product_type} (exact match) for a type in no
    predefined group, or None when the type is unknown/missing (keep all).
    """
    if not product_type:
        return None
    for grp in _PRODUCT_TYPE_GROUPS:
        if product_type in grp:
            return grp
    return {product_type}


def _image_driven_score(img_sim: float) -> float:
    """Map an image similarity into a score contribution.

    Below _IMAGE_MATCH_MIN the images are considered different → 0.0. Above it,
    map [_IMAGE_MATCH_MIN, 1.0] → [_SUSPICIOUS_THRESHOLD, _IMAGE_DRIVEN_CEILING]
    so a near-identical photo guarantees SUSPICIOUS review but never reaches
    DUPLICATE on the image signal alone.
    """
    if img_sim < _IMAGE_MATCH_MIN:
        return 0.0
    span = (img_sim - _IMAGE_MATCH_MIN) / (1.0 - _IMAGE_MATCH_MIN)
    return round(
        _SUSPICIOUS_THRESHOLD + span * (_IMAGE_DRIVEN_CEILING - _SUSPICIOUS_THRESHOLD),
        4,
    )


def _build_result(
    decision: str,
    highest_score: float,
    matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "isDuplicate": decision == "DUPLICATE",
        "highestScore": round(highest_score, 4),
        "decision": decision,
        "suspiciousMatches": [
            {
                "listingId": m.get("listingId"),
                "title": m.get("title", ""),
                "score": round(m.get("score", 0), 4),
                "titleSimilarity": m.get("titleSimilarity", 0),
                "descriptionSimilarity": m.get("descriptionSimilarity", 0),
                "addressSimilarity": m.get("addressSimilarity", 0),
                "priceSimilarity": m.get("priceSimilarity", 0),
                "imageSimilarity": m.get("imageSimilarity", 0),
                "llmScore": m.get("llmScore"),
                "llmReason": m.get("llmReason"),
            }
            for m in matches
            if m.get("score", 0) >= _SUSPICIOUS_THRESHOLD
        ],
    }
