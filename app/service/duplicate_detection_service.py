"""
Duplicate Listing Detection Service — 3-step pipeline.

Step 1: Candidate retrieval — query backend for similar listings (same area, price range)
Step 2: Fast similarity scoring — TF-IDF + fuzzy match (Python, no LLM cost)
Step 3: LLM confirmation — Gemini reviews only suspicious matches (saves cost)

Decision thresholds:
  score > 0.85  → DUPLICATE (auto-reject or flag)
  score 0.6-0.85 → SUSPICIOUS (flag for admin review)
  score < 0.6   → PASS
"""

import logging
from typing import Any, Dict, List, Optional

from app.ai.llm.gateway import get_gateway
from app.ai.text_similarity import (
    batch_tfidf_similarity,
    compute_listing_similarity,
)
from app.core import backend_client
from app.core.config import settings

logger = logging.getLogger(__name__)

_DUPLICATE_THRESHOLD = 0.85
_SUSPICIOUS_THRESHOLD = 0.6
_MAX_CANDIDATES = 50
_MAX_LLM_CHECKS = 3

_LLM_PROMPT = """\
Bạn là chuyên gia phát hiện tin đăng trùng lặp trên nền tảng bất động sản.

So sánh TIN MỚI với TIN CŨ bên dưới. Đánh giá xem chúng có phải là cùng một bất động sản không.

Tiêu chí đánh giá:
- Cùng địa chỉ hoặc vị trí rất gần nhau
- Mô tả giống nhau về nội dung (dù có thay đổi từ ngữ)
- Giá và diện tích tương tự
- Ảnh giống nhau (nếu có mô tả ảnh)

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
        if not candidates:
            logger.info("No candidates found — listing is unique.")
            return _build_result("PASS", 0.0, [])

        logger.info("Step 1: %d candidates retrieved.", len(candidates))

        # ── Step 2: Fast similarity scoring ───────────────────────────
        scored = self._score_candidates(listing, candidates)
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
        """Query backend for listings similar in location, price, and type."""
        price = listing.get("price", 0)
        params: Dict[str, Any] = {
            "productType": listing.get("productType"),
            "provinceCode": listing.get("provinceCode"),
            "size": _MAX_CANDIDATES,
            "postedWithinDays": 60,
        }
        if listing.get("districtId"):
            params["districtId"] = listing["districtId"]
        if price:
            params["minPrice"] = int(price * 0.8)
            params["maxPrice"] = int(price * 1.2)
        if listing.get("provinceCode"):
            params["provinceId"] = listing["provinceCode"]

        try:
            data = await backend_client.search_listings(params)
            if "error" in data:
                logger.warning("Backend search failed: %s", data["error"])
                return []
            return data.get("listings", [])
        except Exception as e:
            logger.error("Candidate retrieval failed: %s", e)
            return []

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
        """Use Gemini to confirm or reject suspicious matches."""
        confirmed = []

        for match in suspicious:
            listing_id = match.get("listingId")
            try:
                # Fetch full detail for the candidate
                detail = await backend_client.get_listing(str(listing_id))
                if "error" in detail:
                    match["llmScore"] = None
                    match["llmReason"] = "Could not fetch listing detail"
                    confirmed.append(match)
                    continue

                prompt = _LLM_PROMPT.format(
                    new_title=new_listing.get("title", ""),
                    new_desc=(new_listing.get("description", ""))[:500],
                    new_price=new_listing.get("price", "N/A"),
                    new_area=new_listing.get("area", "N/A"),
                    new_address=self._extract_address(new_listing),
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
                response = await self._gateway.generate(
                    prompt,
                    model_name=settings.GEMINI_CHAT_MODEL,
                    generation_config={"temperature": 0.1, "max_output_tokens": 256},
                    trace=trace,
                    span_name="duplicate-confirm",
                )

                llm_result = self._parse_llm_response(response)
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

            except Exception as e:
                logger.warning("LLM confirm failed for %s: %s", listing_id, e)
                match["llmScore"] = None
                match["llmReason"] = str(e)

            confirmed.append(match)

        confirmed.sort(key=lambda x: x["score"], reverse=True)
        return confirmed

    @staticmethod
    def _parse_llm_response(response: Any) -> Optional[Dict[str, Any]]:
        """Parse JSON from Gemini response."""
        import json

        try:
            text = response.text.strip()
        except (ValueError, AttributeError):
            return None

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
                "llmScore": m.get("llmScore"),
                "llmReason": m.get("llmReason"),
            }
            for m in matches
            if m.get("score", 0) >= _SUSPICIOUS_THRESHOLD
        ],
    }
