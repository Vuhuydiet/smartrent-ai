"""
RAG Retriever — Phase 1: keyword-based lookup over JSON knowledge base files.

Responsibilities
----------------
1. Load three knowledge-base files once at startup (area_codes, amenities, faq).
2. Given a user query, find relevant entries using keyword matching.
3. Return a compact, formatted context string that the Agent Orchestrator injects
   into the system prompt before the LLM sees the user's message.

Phase 2 (future): replace keyword matching with ChromaDB vector search to handle
semantic similarity (e.g. "chỗ để xe" → parking amenity even without exact keywords).

Context string format
---------------------
The returned string is designed to be appended to the system prompt verbatim, e.g.:

    [LOCATION CODES — use these when calling search_listings]
    Cầu Giấy (Hà Nội): provinceCode="01" districtId=5

    [AMENITIES — use amenityIds when user mentions these]
    WiFi → id=1 | Điều hòa → id=2

    [KNOWLEDGE]
    Q: Tiền đặt cọc thuê nhà thường bao nhiêu?
    A: Thường là 1-3 tháng tiền thuê ...
"""

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_KB_DIR = Path(__file__).parent / "knowledge_base"

# How many FAQ entries to include per query at most
_MAX_FAQ = 3
# How many platform guide entries to include per query at most
_MAX_GUIDES = 1
# How many amenity matches to include
_MAX_AMENITIES = 8
# Minimum keyword match score to include an FAQ entry
_FAQ_MIN_SCORE = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_json(filename: str) -> Any:
    path = _KB_DIR / filename
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Knowledge base file not found: %s", path)
        return {}
    except json.JSONDecodeError as e:
        logger.error("Malformed JSON in %s: %s", filename, e)
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """
    Lowercase, strip whitespace, and remove diacritics so that
    "dat coc" matches "đặt cọc" and "wifi" matches "WiFi".
    """
    nfkd = unicodedata.normalize("NFKD", text.lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Synonym expansion — maps normalised terms to semantic equivalents so that
# "tien phai tra truoc" matches the "dat coc" FAQ, etc.
# ---------------------------------------------------------------------------

_SYNONYMS: Dict[str, List[str]] = {
    "dat coc": ["tien phai tra truoc", "tien tam ung", "prepay", "tien giu cho"],
    "hop dong": ["contract", "lease", "ky ket", "giay to thue"],
    "vip": ["goi tin", "nang cap", "subscription", "goi dich vu", "membership"],
    "dang tin": ["tao tin", "post listing", "dang bai", "cho thue phong"],
    "lien he": ["goi dien", "nhan tin", "contact", "so dien thoai", "zalo"],
    "doi mat khau": [
        "thay mat khau",
        "change password",
        "reset password",
        "quen mat khau",
    ],
    "xoa tai khoan": ["huy tai khoan", "delete account", "dong tai khoan"],
    "luu tin": ["save", "bookmark", "yeu thich", "quan tam"],
    "bao cao": ["report", "to cao", "tin gia", "lua dao", "vi pham", "khieu nai"],
    "gia han": ["renew", "dang lai", "gia han tin", "gia han vip"],
    "goi y": ["de xuat", "recommendation", "tu van", "phu hop"],
    "an toan": ["safety", "can than", "phong tranh", "bao ve"],
    "thong bao": ["notification", "alert", "canh bao", "nhac nho"],
    "chia se": ["share", "gui tin", "gui link"],
    "bo loc": ["filter", "loc tim kiem", "tim kiem nang cao"],
    "thanh toan": ["payment", "nap tien", "chuyen tien", "phi"],
    "xac minh": ["verify", "xac thuc", "otp"],
    "tranh chap": ["dispute", "khieu nai", "giai quyet", "phan nan"],
}


def _keyword_score(query_norm: str, keywords: List[str]) -> int:
    """
    Count how many keywords (normalised) appear in the normalised query.
    Also checks synonym groups: if a keyword belongs to a synonym group and
    any synonym from that group appears in the query, it counts as a match.
    """
    score = 0
    for kw in keywords:
        kw_norm = _normalise(kw)
        if kw_norm in query_norm:
            score += 1
            continue
        # Check if any synonym of this keyword appears in the query
        for _canonical, synonyms in _SYNONYMS.items():
            group = [_canonical] + synonyms
            if kw_norm in group and any(syn in query_norm for syn in group):
                score += 1
                break
    return score


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------


class RAGRetriever:
    """
    Keyword-based retriever over three JSON knowledge-base files.

    All data is loaded eagerly at construction time so per-request retrieval
    is pure CPU (no I/O).
    """

    def __init__(self) -> None:
        raw_areas = _load_json("area_codes.json")
        raw_amenities = _load_json("amenities.json")
        raw_faq = _load_json("faq.json")
        raw_guides = _load_json("platform_guide.json")

        self._provinces: List[Dict] = raw_areas.get("provinces", [])
        self._districts: Dict[str, List[Dict]] = raw_areas.get("districts", {})
        self._amenities: List[Dict] = raw_amenities.get("amenities", [])
        self._faq: List[Dict] = raw_faq.get("entries", [])
        self._guides: List[Dict] = raw_guides.get("features", [])

        logger.info(
            "RAGRetriever loaded: %d provinces, %d amenities, %d FAQ entries, %d guides",
            len(self._provinces),
            len(self._amenities),
            len(self._faq),
            len(self._guides),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> str:
        """
        Given a user query, return a context string to inject into the
        system prompt.  Returns an empty string when nothing relevant is found.

        Normalisation (lowercase + diacritic strip) is applied once here so
        all internal matchers work on the same representation.
        """
        query_norm = _normalise(query)
        parts: List[str] = []

        location_ctx = self._location_context(query_norm)
        if location_ctx:
            parts.append(location_ctx)

        amenity_ctx = self._amenity_context(query_norm)
        if amenity_ctx:
            parts.append(amenity_ctx)

        faq_ctx = self._faq_context(query_norm)
        if faq_ctx:
            parts.append(faq_ctx)

        guide_ctx = self._guide_context(query_norm)
        if guide_ctx:
            parts.append(guide_ctx)

        return "\n\n".join(parts)

    def get_province_code(self, name: str) -> Optional[str]:
        """Resolve a province/city name to its code. Returns None if unknown."""
        name_norm = _normalise(name)
        for p in self._provinces:
            if name_norm == _normalise(p["name"]) or name_norm in [
                _normalise(a) for a in p.get("aliases", [])
            ]:
                return p["code"]
        return None

    def get_district_code(self, province_code: str, name: str) -> Optional[str]:
        """Resolve a district name within a province to its code."""
        name_norm = _normalise(name)
        for d in self._districts.get(province_code, []):
            if name_norm == _normalise(d["name"]) or name_norm in [
                _normalise(a) for a in d.get("aliases", [])
            ]:
                return d["code"]
        return None

    def get_amenity_ids(self, names: List[str]) -> List[int]:
        """Resolve a list of amenity name strings to their IDs."""
        ids: List[int] = []
        for name in names:
            name_norm = _normalise(name)
            for amenity in self._amenities:
                all_names = [amenity["name"]] + amenity.get("aliases", [])
                if any(name_norm in _normalise(a) for a in all_names):
                    ids.append(amenity["id"])
                    break
        return ids

    def get_system_prompt_prefix(self) -> str:
        """
        Return a static block injected into the system prompt once at startup.
        It tells the model about all available province codes and common amenity IDs
        so it can use them without requiring per-query retrieval.
        """
        lines: List[str] = []

        # Province codes
        lines.append("MÃ TỈNH/THÀNH PHỐ (dùng cho tham số provinceCode):")
        for p in self._provinces:
            aliases = ", ".join(p.get("aliases", []))
            lines.append(
                f'  {p["name"]} → provinceCode="{p["code"]}" (aliases: {aliases})'
            )

        lines.append("")
        lines.append("MÃ QUẬN/HUYỆN (dùng cho tham số districtId — kiểu INTEGER):")
        for prov_code, districts in self._districts.items():
            prov_name = next(
                (p["name"] for p in self._provinces if p["code"] == prov_code),
                prov_code,
            )
            district_strs = [f'{d["name"]}={int(d["code"])}' for d in districts]
            lines.append(f"  {prov_name}: {', '.join(district_strs)}")

        lines.append("")
        lines.append("MÃ TIỆN NGHI PHỔ BIẾN (dùng cho amenityIds):")
        for a in self._amenities[:10]:  # top 10 most common
            aliases = ", ".join(a.get("aliases", [])[:3])
            lines.append(f'  {a["name"]} → id={a["id"]} (aliases: {aliases})')

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private — location matching
    # ------------------------------------------------------------------

    def _location_context(self, query_norm: str) -> str:
        """
        Detect mentioned province/district names and return resolved codes.
        Example output:
            [MÃ ĐỊA ĐIỂM]
            Cầu Giấy (Hà Nội): cityCode="01" districtCode="005"
        """
        matched: List[
            Tuple[str, str, str, str]
        ] = []  # (province_name, district_name, province_code, district_code)

        for province in self._provinces:
            province_names = [province["name"]] + province.get("aliases", [])
            province_mentioned = any(
                _normalise(n) in query_norm for n in province_names
            )

            for district in self._districts.get(province["code"], []):
                district_names = [district["name"]] + district.get("aliases", [])
                district_mentioned = any(
                    _normalise(n) in query_norm for n in district_names
                )

                if district_mentioned:
                    matched.append(
                        (
                            province["name"],
                            district["name"],
                            province["code"],
                            district["code"],
                        )
                    )
                elif province_mentioned and not matched:
                    # Province mentioned but no specific district — only emit city code
                    matched.append((province["name"], "", province["code"], ""))

        if not matched:
            return ""

        lines = ["[MÃ ĐỊA ĐIỂM — dùng các giá trị này khi gọi search_listings]"]
        seen_provinces: set = set()

        for prov_name, dist_name, prov_code, dist_code in matched:
            if dist_name:
                lines.append(
                    f'  {dist_name} ({prov_name}): provinceCode="{prov_code}" districtId={int(dist_code)}'
                )
            elif prov_name not in seen_provinces:
                lines.append(f'  {prov_name}: provinceCode="{prov_code}"')
                seen_provinces.add(prov_name)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private — amenity matching
    # ------------------------------------------------------------------

    def _amenity_context(self, query_norm: str) -> str:
        """
        Detect mentioned amenities and return their IDs.
        Example output:
            [TIỆN NGHI]
            WiFi → amenityId=1 | Điều hòa → amenityId=2
        """
        matched: List[str] = []
        for amenity in self._amenities:
            all_names = [amenity["name"]] + amenity.get("aliases", [])
            if any(_normalise(n) in query_norm for n in all_names):
                matched.append(f'{amenity["name"]} → amenityId={amenity["id"]}')
                if len(matched) >= _MAX_AMENITIES:
                    break

        if not matched:
            return ""

        return "[TIỆN NGHI — dùng amenityIds khi gọi search_listings]\n  " + " | ".join(
            matched
        )

    # ------------------------------------------------------------------
    # Private — FAQ matching
    # ------------------------------------------------------------------

    def _faq_context(self, query_norm: str) -> str:
        """
        Find the most relevant FAQ entries and return them as Q&A pairs.
        """
        scored: List[Tuple[int, Dict]] = []
        for entry in self._faq:
            score = _keyword_score(query_norm, entry.get("keywords", []))
            if score >= _FAQ_MIN_SCORE:
                scored.append((score, entry))

        if not scored:
            return ""

        # Sort by score descending, take top N
        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by category — prefer highest-scoring entry per category,
        # but allow a second entry from the same category only if we haven't
        # filled _MAX_FAQ slots with unique categories yet.
        seen_categories: set = set()
        top: List[Tuple[int, Dict]] = []
        deferred: List[Tuple[int, Dict]] = []
        for item in scored:
            cat = item[1].get("category", "")
            if cat not in seen_categories:
                top.append(item)
                seen_categories.add(cat)
            else:
                deferred.append(item)
            if len(top) >= _MAX_FAQ:
                break
        # Fill remaining slots with deferred (same-category) entries
        for item in deferred:
            if len(top) >= _MAX_FAQ:
                break
            top.append(item)

        lines = ["[THÔNG TIN THAM KHẢO]"]
        for _, entry in top:
            lines.append(f'  Q: {entry["question"]}')
            lines.append(f'  A: {entry["answer"]}')

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private — Platform guide matching
    # ------------------------------------------------------------------

    def _guide_context(self, query_norm: str) -> str:
        """
        Find the most relevant platform guide and return step-by-step instructions.
        Returns at most _MAX_GUIDES entries to keep the system prompt manageable.
        """
        scored: List[Tuple[int, Dict]] = []
        for guide in self._guides:
            score = _keyword_score(query_norm, guide.get("keywords", []))
            if score >= _FAQ_MIN_SCORE:
                scored.append((score, guide))

        if not scored:
            return ""

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:_MAX_GUIDES]

        lines = ["[HƯỚNG DẪN SỬ DỤNG]"]
        for _, guide in top:
            lines.append(f'  {guide["name"]}:')
            for i, step in enumerate(guide.get("steps", []), 1):
                lines.append(f"    Bước {i}: {step}")
            tips = guide.get("tips", [])
            if tips:
                lines.append("  Mẹo:")
                for tip in tips[:2]:
                    lines.append(f"    - {tip}")

        return "\n".join(lines)
