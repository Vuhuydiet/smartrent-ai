"""
Deterministic free-text → structured filter resolver.

This "borrows" the AI chatbox context: it resolves a Vietnamese rental search
query into backend-ready filters using the SAME knowledge base the chatbox's
``search_listings`` tool relies on (:class:`RAGRetriever` — provinces,
districts, amenities). No LLM round-trip, so it is cheap enough to run on every
``/search/parse`` and ``/search/suggestions`` call and is fully deterministic.

Why it exists
-------------
The old ``/search/parse`` returned loose text (``district="tân bình"``,
``amenities=["máy lạnh"]``) and, on any failure, dumped the WHOLE query into
``keyword``. The backend then FULLTEXT-ANDed every token ("dưới", "5tr", …),
matching zero listings. Here we instead resolve:

  * "tân bình"  → provinceCode 79 / legacyDistrictId 766   (RAG area_codes)
  * "máy lạnh"  → amenityIds [2]                            (RAG amenities)
  * "trọ"       → productTypes ["ROOM", "APARTMENT"]        (chatbox rule)
  * "dưới 5tr"  → maxPrice 5_000_000

Price / area / bedroom extraction mirrors the Java ``SearchQueryParser`` regex
so the AI service and the backend's local fallback agree on the same numbers.
"""

import re
import unicodedata
from typing import List, Optional, Tuple

from app.agent.rag.retriever import RAGRetriever, _normalise
from app.dto.search import AiParsedCriteriaDto, AppliedFilters

# Lazily-built singleton — the RAG knowledge base is a few small JSON files but
# re-reading them per request still has measurable cost. Mirrors the pattern in
# app/agent/tools/address_translator.py.
_rag_singleton: Optional[RAGRetriever] = None


def _rag() -> RAGRetriever:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RAGRetriever()
    return _rag_singleton


# ---------------------------------------------------------------------------
# Normalisation / abbreviation expansion (subset of Java expandAbbreviations)
# ---------------------------------------------------------------------------


def _strip(text: Optional[str]) -> str:
    """Lowercase, drop đ, strip diacritics & punctuation, collapse spaces."""
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def _expand(norm: str) -> str:
    """Expand the abbreviations VN users type so regex/lookup see one spelling."""
    r = f" {norm} "
    pairs = {
        " canho ": " can ho ",
        " chungcu ": " chung cu ",
        " phongtro ": " phong tro ",
        " nhatro ": " nha tro ",
        " dhqg ": " dai hoc quoc gia ",
        " maylanh ": " may lanh ",
        " may lan ": " may lanh ",
        " dieuhoa ": " dieu hoa ",
        " mlanh ": " may lanh ",
        " mgiat ": " may giat ",
        " tlanh ": " tu lanh ",
        " full nt ": " full noi that ",
        " fullnt ": " full noi that ",
        " tphcm ": " ho chi minh ",
        " hcm ": " ho chi minh ",
        " sg ": " ho chi minh ",
        " hn ": " ha noi ",
    }
    for src, dst in pairs.items():
        r = r.replace(src, dst)
    for i in range(1, 13):
        r = r.replace(f" q{i} ", f" quan {i} ")
    r = re.sub(r"(\d+)\s*pn\b", r"\1 phong ngu", r)
    r = re.sub(r"(\d+)\s*p\s*ngu\b", r"\1 phong ngu", r)
    r = re.sub(r"(\d+(?:[.,]\d+)?)\s*m\s*2\b", r"\1 m2", r)
    r = re.sub(r"(\d+(?:[.,]\d+)?)\s*(?:met vuong|met vng|mv)\b", r"\1 m2", r)
    r = re.sub(r"(\d+)\s*tr\b", r"\1 trieu", r)
    r = re.sub(r"(\d+)\s*ty\b", r"\1 ty", r)
    r = re.sub(r"(\d+)\s*k\b", r"\1 nghin", r)
    return re.sub(r"\s+", " ", r).strip()


# ---------------------------------------------------------------------------
# Price / area / bedrooms — patterns ported from Java SearchQueryParser
# ---------------------------------------------------------------------------

_UNIT = r"(trieu|tr|ty|nghin|ngan|k|cu)"
_RANGE_RE = re.compile(
    rf"(?:tu\s+)?(\d+(?:[.,]\d+)?)\s*{_UNIT}?\s*(?:-|den|toi)\s*(\d+(?:[.,]\d+)?)\s*{_UNIT}"
)
_MAX_RE = re.compile(rf"(?:duoi|khong qua|toi da|<=|<)\s*(\d+(?:[.,]\d+)?)\s*{_UNIT}?")
_MIN_RE = re.compile(rf"(?:tren|tu|it nhat|>=|>)\s*(\d+(?:[.,]\d+)?)\s*{_UNIT}?")
_LONE_PRICE_RE = re.compile(rf"(\d+(?:[.,]\d+)?)\s*{_UNIT}")

_AREA_RANGE_RE = re.compile(
    r"(?:dien tich\s+)?(?:tu\s+)?(\d+(?:[.,]\d+)?)\s*(?:m2)?\s*(?:-|den|toi)\s*(\d+(?:[.,]\d+)?)\s*m2"
)
_AREA_MAX_RE = re.compile(r"(?:duoi|nho hon|toi da|<=|<)\s*(\d+(?:[.,]\d+)?)\s*m2")
_AREA_MIN_RE = re.compile(
    r"(?:tren|tu|lon hon|rong tren|rong hon|>=|>)\s*(\d+(?:[.,]\d+)?)\s*m2"
)
_AREA_LONE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*m2")
_BEDROOM_RE = re.compile(r"(\d+)\s*phong ngu")


def _money(number: str, unit: Optional[str]) -> Optional[float]:
    try:
        value = float(number.replace(",", "."))
    except ValueError:
        return None
    mult = {
        "ty": 1_000_000_000,
        "trieu": 1_000_000,
        "tr": 1_000_000,
        "cu": 1_000_000,
        "nghin": 1_000,
        "ngan": 1_000,
        "k": 1_000,
    }.get(
        unit or "", 1_000_000
    )  # bare number in a rental context ≈ triệu
    return value * mult


def _to_float(number: str) -> Optional[float]:
    try:
        return float(number.replace(",", "."))
    except ValueError:
        return None


def _extract_bedrooms(working: str) -> Tuple[str, Optional[int]]:
    m = _BEDROOM_RE.search(working)
    bedrooms: Optional[int] = None
    if m:
        try:
            bedrooms = int(m.group(1))
        except ValueError:
            bedrooms = None
        working = working.replace(m.group(0), " ")
    return working.replace(" phong ngu ", " "), bedrooms


def _extract_area(working: str) -> Tuple[str, Optional[float], Optional[float]]:
    min_a: Optional[float] = None
    max_a: Optional[float] = None
    rng = _AREA_RANGE_RE.search(working)
    if rng:
        min_a = _to_float(rng.group(1))
        max_a = _to_float(rng.group(2))
        return working.replace(rng.group(0), " "), min_a, max_a
    mx = _AREA_MAX_RE.search(working)
    if mx:
        max_a = _to_float(mx.group(1))
        working = working.replace(mx.group(0), " ")
    mn = _AREA_MIN_RE.search(working)
    if mn:
        min_a = _to_float(mn.group(1))
        working = working.replace(mn.group(0), " ")
    if min_a is None and max_a is None:
        lone = _AREA_LONE_RE.search(working)
        if lone:
            min_a = _to_float(lone.group(1))
            working = working.replace(lone.group(0), " ")
    return working.replace(" m2 ", " "), min_a, max_a


def _extract_price(working: str) -> Tuple[str, Optional[float], Optional[float]]:
    min_p: Optional[float] = None
    max_p: Optional[float] = None
    rng = _RANGE_RE.search(working)
    if rng:
        unit = rng.group(4) or rng.group(2)
        min_p = _money(rng.group(1), unit)
        max_p = _money(rng.group(3), unit)
        return working.replace(rng.group(0), " "), min_p, max_p
    mx = _MAX_RE.search(working)
    if mx:
        max_p = _money(mx.group(1), mx.group(2))
        working = working.replace(mx.group(0), " ")
    mn = _MIN_RE.search(working)
    if mn:
        min_p = _money(mn.group(1), mn.group(2))
        working = working.replace(mn.group(0), " ")
    if min_p is None and max_p is None:
        lone = _LONE_PRICE_RE.search(working)
        if lone:
            # bare "5 triệu" with no comparator → treat as an upper bound
            max_p = _money(lone.group(1), lone.group(2))
            working = working.replace(lone.group(0), " ")
    return working, min_p, max_p


# ---------------------------------------------------------------------------
# Property type — chatbox ambiguity rule (orchestrator _SYSTEM_BASE)
# ---------------------------------------------------------------------------

# Ordered: longer / more specific phrases first. Value = list of backend enums.
# Ambiguous VN terms span multiple types exactly like the chatbox does.
_PRODUCT_TYPES: List[Tuple[str, List[str]]] = [
    ("phong tro", ["ROOM"]),
    ("phong don", ["ROOM"]),
    ("phong cho thue", ["ROOM"]),
    ("can ho dich vu", ["APARTMENT"]),
    ("can ho", ["APARTMENT"]),
    ("chung cu", ["APARTMENT"]),
    ("nha nguyen can", ["HOUSE"]),
    ("nha rieng", ["HOUSE"]),
    ("nha tro", ["ROOM", "APARTMENT"]),
    ("van phong", ["OFFICE"]),
    ("mat bang", ["OFFICE"]),
    ("studio", ["STUDIO"]),
    ("thue nha", ["ROOM", "APARTMENT", "HOUSE"]),
    ("tim nha", ["ROOM", "APARTMENT", "HOUSE"]),
    ("tro", ["ROOM", "APARTMENT"]),
    ("nha", ["HOUSE"]),
]

_LISTING_TYPES: List[Tuple[str, str]] = [
    ("o ghep", "SHARE"),
    ("can ban", "SALE"),
    ("ban", "SALE"),
    ("can thue", "RENT"),
    ("cho thue", "RENT"),
    ("thue", "RENT"),
]

# Tokens that are never a location and carry no filter on their own — used to
# decide what (if anything) is left as a residual keyword.
_STOPWORDS = {
    "gan",
    "o",
    "tai",
    "khu",
    "vuc",
    "vung",
    "quanh",
    "gia",
    "re",
    "dep",
    "moi",
    "rong",
    "rai",
    "co",
    "va",
    "can",
    "tim",
    "thue",
    "muon",
    "mua",
    "phong",
    "nha",
    "cho",
    "duoi",
    "tren",
    "tu",
    "den",
    "khoang",
    "tam",
    "khong",
    "qua",
    "toi",
    "da",
    "it",
    "nhat",
    "dien",
    "tich",
    "m2",
    "met",
    "vuong",
    "mv",
    "ngu",
    "pn",
    "lon",
    "nho",
    "hon",
}


def _match_product_types(working: str) -> Tuple[str, List[str]]:
    for phrase, enums in _PRODUCT_TYPES:
        if f" {phrase} " in working:
            return working.replace(f" {phrase} ", " "), list(enums)
    return working, []


def _match_listing_type(working: str) -> Tuple[str, Optional[str]]:
    for phrase, enum in _LISTING_TYPES:
        if f" {phrase} " in working:
            return working.replace(f" {phrase} ", " "), enum
    return working, None


# ---------------------------------------------------------------------------
# Location — scan RAG provinces/districts (same KB the chatbox uses)
# ---------------------------------------------------------------------------


def _resolve_location(
    query_norm: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Find the most specific (province, district) the query mentions.

    Returns (provinceCode, districtCode, matched_norm_phrases). District match
    implies the province. Mirrors RAGRetriever._location_context but returns
    structured codes instead of a prompt string.
    """
    rag = _rag()
    matched_phrases: List[str] = []

    province_code: Optional[str] = None
    district_code: Optional[str] = None

    for province in rag._provinces:
        prov_names = [province["name"]] + province.get("aliases", [])
        prov_hit = next(
            (n for n in prov_names if _normalise(n) and _normalise(n) in query_norm),
            None,
        )
        for district in rag._districts.get(province["code"], []):
            dist_names = [district["name"]] + district.get("aliases", [])
            dist_hit = next(
                (
                    n
                    for n in dist_names
                    if _normalise(n) and _normalise(n) in query_norm
                ),
                None,
            )
            if dist_hit:
                province_code = province["code"]
                district_code = district["code"]
                matched_phrases.append(_normalise(dist_hit))
                if prov_hit:
                    matched_phrases.append(_normalise(prov_hit))
                return province_code, district_code, matched_phrases

        if prov_hit and province_code is None:
            province_code = province["code"]
            matched_phrases.append(_normalise(prov_hit))

    return province_code, district_code, matched_phrases


def _resolve_amenities(
    query_norm: str,
) -> Tuple[List[int], List[str], List[str]]:
    """Return (amenityIds, display names, matched norm phrases)."""
    rag = _rag()
    ids: List[int] = []
    names: List[str] = []
    matched: List[str] = []
    for amenity in rag._amenities:
        all_names = [amenity["name"]] + amenity.get("aliases", [])
        hit = next(
            (n for n in all_names if _normalise(n) and _normalise(n) in query_norm),
            None,
        )
        if hit and amenity["id"] not in ids:
            ids.append(amenity["id"])
            names.append(amenity["name"])
            matched.append(_normalise(hit))
    return ids, names, matched


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _as_int(code: Optional[str]) -> Optional[int]:
    if code is None:
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


def resolve_applied_filters(
    query: str,
    criteria: Optional[AiParsedCriteriaDto] = None,
) -> Optional[AppliedFilters]:
    """
    Resolve a free-text query into :class:`AppliedFilters`.

    When ``criteria`` (an LLM parse) is supplied, its explicit numeric/type
    values win; names (province/district/amenities) are still resolved to ids
    via the RAG knowledge base. Returns ``None`` only when nothing structured
    could be extracted at all (caller may then keep a plain keyword search).
    """
    norm = _expand(_strip(query))
    if not norm:
        return None

    query_norm = f" {norm} "
    working = query_norm

    working, bedrooms = _extract_bedrooms(working)
    working, min_area, max_area = _extract_area(working)
    working, min_price, max_price = _extract_price(working)
    working, product_types = _match_product_types(working)
    working, listing_type = _match_listing_type(working)

    province_code, district_code, loc_phrases = _resolve_location(query_norm)
    amenity_ids, amenity_names, amen_phrases = _resolve_amenities(query_norm)

    # Overlay explicit values the LLM parsed (it wins for numbers/type/listing).
    if criteria is not None:
        if criteria.minPrice is not None:
            min_price = criteria.minPrice
        if criteria.maxPrice is not None:
            max_price = criteria.maxPrice
        if criteria.minArea is not None:
            min_area = criteria.minArea
        if criteria.maxArea is not None:
            max_area = criteria.maxArea
        if criteria.bedrooms is not None:
            bedrooms = criteria.bedrooms
        if criteria.listingType:
            listing_type = criteria.listingType.upper()
        if criteria.propertyType and not product_types:
            product_types = [criteria.propertyType.upper()]
        # Resolve any location/amenity NAMES the LLM gave but the raw scan
        # missed (e.g. it expanded an abbreviation we don't know).
        if province_code is None:
            for name in (criteria.district, criteria.ward, criteria.province):
                if not name:
                    continue
                pc, dc, _ = _resolve_location(f" {_strip(name)} ")
                if pc:
                    province_code, district_code = pc, dc or district_code
                    break
        if not amenity_ids and criteria.amenities:
            extra_ids = _rag().get_amenity_ids(criteria.amenities)
            if extra_ids:
                amenity_ids = extra_ids

    # Build the residual keyword from genuinely-unconsumed content tokens. We
    # deliberately do NOT echo the whole query back — that is the bug.
    consumed = set()
    for phrase in loc_phrases + amen_phrases:
        consumed.update(phrase.split())
    residual_tokens = [
        tok
        for tok in working.split()
        if len(tok) >= 2
        and not tok.isdigit()
        and tok not in _STOPWORDS
        and tok not in consumed
    ]
    residual = " ".join(residual_tokens).strip() or None

    has_other_structured = bool(
        product_types
        or listing_type
        or min_price is not None
        or max_price is not None
        or min_area is not None
        or max_area is not None
        or bedrooms is not None
        or amenity_ids
    )
    # Promote leftover text to `locationText` ONLY when the location didn't
    # resolve to ids AND the query carried other structured intent — then the
    # leftover is almost certainly the place we couldn't map. With no other
    # signal it's just a free-text query, so keep it as a plain `keyword`
    # (a bare keyword must never become a bogus district filter downstream).
    location_text: Optional[str] = None
    if province_code is None and residual and has_other_structured:
        location_text, residual = residual, None

    af = AppliedFilters(
        productType=product_types[0] if product_types else None,
        productTypes=product_types,
        listingType=listing_type,
        minPrice=min_price,
        maxPrice=max_price,
        minArea=min_area,
        maxArea=max_area,
        bedrooms=bedrooms,
        provinceCode=province_code,
        districtCode=district_code,
        legacyProvinceId=_as_int(province_code),
        legacyDistrictId=_as_int(district_code),
        amenityIds=amenity_ids,
        amenities=amenity_names,
        amenityMatchMode="ALL" if amenity_ids else None,
        locationText=location_text,
        keyword=residual,
    )

    # Nothing structured at all → let the caller decide (keyword search).
    if not (
        af.productTypes
        or af.listingType
        or af.minPrice is not None
        or af.maxPrice is not None
        or af.minArea is not None
        or af.maxArea is not None
        or af.bedrooms is not None
        or af.provinceCode
        or af.amenityIds
        or af.locationText
        or af.keyword
    ):
        return None
    return af
