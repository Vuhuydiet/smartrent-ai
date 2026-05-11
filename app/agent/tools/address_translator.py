"""
Tool: address_translator — bridges Vietnam's pre-2025-07 (3-tier) and
post-reform (2-tier) administrative structures.

Triggered by user phrases like "Quận Bình Thạnh giờ là phường nào?",
"Phường 1 thuộc quận cũ nào?", "tra cứu địa chỉ X".

Strategy:
  1. Local lookup in RAG `area_codes.json` for LEGACY province + district
     match (covers the common case: user names a familiar district).
  2. Backend `/v1/addresses/search-new-address` for the NEW-structure ward
     matches.
  3. Return both halves so the LLM can explain the mapping.
"""

import logging
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.rag.retriever import RAGRetriever, _normalise
from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

_MAX_NEW_RESULTS = 10

# Lazily-built singleton — RAG knowledge base is cheap (3 small JSON files)
# but allocating a new one per tool call still has measurable overhead.
_rag_singleton: Optional[RAGRetriever] = None


def _rag() -> RAGRetriever:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAGRetriever()
    return _rag_singleton


# Substring matches under this length are too noisy — "1" or "q" would
# silently pick the first district whose normalised alias happens to
# contain that character. We require ≥4 chars before allowing substring
# matching, otherwise demand an exact whole-string match.
_SUBSTRING_MATCH_MIN_LEN = 4


def _legacy_match(query: str) -> Optional[Dict[str, Any]]:
    """
    Match `query` against legacy provinces/districts in the RAG knowledge base.

    Two-pass strategy to avoid ambiguous matches: first sweep the whole
    catalogue for an exact normalised match; only if nothing matches
    exactly do we fall back to substring matching, and only for queries
    of meaningful length. This prevents one-character queries like "1"
    or "q" from silently resolving to whichever district happened to
    appear first in the iteration order.

    Districts are preferred over provinces when both match.
    """
    qn = _normalise(query)
    if not qn:
        logger.debug("address_translator: query %r normalises to empty", query)
        return None

    rag = _rag()

    # Build (level, payload, candidates) iterable to traverse twice.
    def _districts_iter() -> Any:
        for prov_code, districts in rag._districts.items():
            prov_name = next(
                (p["name"] for p in rag._provinces if p["code"] == prov_code),
                prov_code,
            )
            for d in districts:
                payload = {
                    "level": "district",
                    "name": d["name"],
                    "provinceName": prov_name,
                    "provinceCode": prov_code,
                    "districtId": int(d["code"]),
                }
                candidates = [_normalise(d["name"])] + [
                    _normalise(a) for a in d.get("aliases", [])
                ]
                yield payload, candidates

    def _provinces_iter() -> Any:
        for p in rag._provinces:
            payload = {
                "level": "province",
                "name": p["name"],
                "provinceName": p["name"],
                "provinceCode": p["code"],
            }
            candidates = [_normalise(p["name"])] + [
                _normalise(a) for a in p.get("aliases", [])
            ]
            yield payload, candidates

    # Pass 1: exact match on districts, then provinces.
    for payload, candidates in _districts_iter():
        if qn in candidates:
            return payload
    for payload, candidates in _provinces_iter():
        if qn in candidates:
            return payload

    # Pass 2: substring match, only if query is long enough to be
    # meaningful. Shorter queries are rejected to avoid false positives.
    if len(qn) < _SUBSTRING_MATCH_MIN_LEN:
        return None

    for payload, candidates in _districts_iter():
        if any(qn in c for c in candidates):
            return payload
    for payload, candidates in _provinces_iter():
        if any(qn in c for c in candidates):
            return payload

    return None


async def _do_translate(query: str) -> Dict[str, Any]:
    """Core translation logic — separated so it can be called directly in tests."""
    legacy = _legacy_match(query)

    new_matches: List[Dict[str, Any]] = []
    try:
        new_data = await backend_client.search_new_address(
            keyword=query, page=1, limit=_MAX_NEW_RESULTS
        )
        if "error" not in new_data:
            items = (
                new_data.get("addresses")
                or new_data.get("content")
                or new_data.get("results")
                or []
            )
            for item in items[:_MAX_NEW_RESULTS]:
                new_matches.append(
                    {
                        "provinceCode": item.get("provinceCode")
                        or item.get("newProvinceCode"),
                        "provinceName": item.get("provinceName")
                        or item.get("newProvinceName"),
                        "wardCode": item.get("wardCode") or item.get("newWardCode"),
                        "wardName": item.get("wardName") or item.get("newWardName"),
                    }
                )
    except httpx.HTTPStatusError as e:
        logger.warning(
            "address_translator: search_new_address HTTP %s",
            e.response.status_code,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("address_translator: search_new_address failed: %s", e)

    if not legacy and not new_matches:
        return {
            "status": "success",
            "query": query,
            "legacy": None,
            "newMatches": [],
            "note": (
                "Không tìm thấy địa chỉ phù hợp ở cả cấu trúc cũ lẫn mới. "
                "Có thể tên chưa chính xác hoặc địa chỉ không có trong "
                "danh mục hệ thống."
            ),
        }

    return {
        "status": "success",
        "query": query,
        "legacy": legacy,
        "newMatches": new_matches,
        "newMatchCount": len(new_matches),
    }


@function_tool(
    name_override="address_translator",
    description_override=(
        "Translate a Vietnamese place name between the legacy (pre-2025-07, "
        "3-tier province→district→ward) and the new (post-reform, 2-tier "
        "province→ward) administrative structures. Use when the user asks "
        "'quận X giờ là phường nào', 'phường Y thuộc quận cũ nào', or when "
        "YOU need to disambiguate a place name before searching listings. "
        "Returns: legacy info (if matched in RAG) + a list of NEW-structure "
        "ward matches from the backend address registry. Explain that one "
        "old district often spans multiple new wards."
    ),
)
async def address_translator(
    ctx: RunContextWrapper[ToolContext],
    query: Annotated[
        str,
        Field(
            description=(
                "The place name to translate. Can be a Vietnamese district "
                "name ('Bình Thạnh', 'Quận 1', 'Cầu Giấy'), a new ward name "
                "('Phường 1', 'Phường Hồng Hà'), or free text. Diacritics "
                "optional."
            )
        ),
    ],
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        return {"status": "error", "error": "Tham số `query` rỗng."}
    return await _do_translate(q)
