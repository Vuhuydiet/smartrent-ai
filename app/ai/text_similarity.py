"""
Text similarity utilities for duplicate listing detection.

Provides fast, offline similarity scoring between listing texts using:
- TF-IDF cosine similarity (captures overall content overlap)
- Fuzzy token-sort ratio (captures title reordering / minor edits)
- Normalized Vietnamese text comparison

These are used as Step 2 (fast filter) before the expensive LLM confirmation
in the duplicate detection pipeline.
"""

import re
import unicodedata
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def normalize_vietnamese(text: str) -> str:
    """
    Normalize Vietnamese text for comparison.
    Lowercase, collapse whitespace, strip diacritics for fuzzy matching.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics: 'phòng trọ' → 'phong tro'."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def fuzzy_ratio(a: str, b: str) -> float:
    """
    Token-sort fuzzy ratio between two strings (0.0 to 1.0).
    Handles word reordering: 'phòng trọ giá rẻ' ≈ 'giá rẻ phòng trọ'.

    Uses a simple token-based approach that doesn't require rapidfuzz.
    If rapidfuzz is available, uses it for better accuracy.
    """
    try:
        from rapidfuzz import fuzz  # type: ignore[import]

        return fuzz.token_sort_ratio(a, b) / 100.0
    except ImportError:
        pass

    # Fallback: Jaccard similarity on token sets
    tokens_a = set(strip_diacritics(normalize_vietnamese(a)).split())
    tokens_b = set(strip_diacritics(normalize_vietnamese(b)).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    TF-IDF cosine similarity between two texts (0.0 to 1.0).
    Good for comparing longer texts like listing descriptions.
    """
    texts = [normalize_vietnamese(text_a), normalize_vietnamese(text_b)]
    if not texts[0] or not texts[1]:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),  # character n-grams work well for Vietnamese
        )
        matrix = vectorizer.fit_transform(texts)
        score = sklearn_cosine(matrix[0:1], matrix[1:2])[0][0]
        return float(score)
    except ValueError:
        return 0.0


def batch_tfidf_similarity(query_text: str, candidate_texts: List[str]) -> List[float]:
    """
    Compute TF-IDF cosine similarity between a query and multiple candidates.
    More efficient than calling tfidf_cosine_similarity() in a loop.
    """
    if not candidate_texts:
        return []

    all_texts = [normalize_vietnamese(query_text)] + [
        normalize_vietnamese(t) for t in candidate_texts
    ]

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
        )
        matrix = vectorizer.fit_transform(all_texts)
        scores = sklearn_cosine(matrix[0:1], matrix[1:])[0]
        return [float(s) for s in scores]
    except ValueError:
        return [0.0] * len(candidate_texts)


def compute_listing_similarity(
    new_listing: dict,
    candidate: dict,
    desc_tfidf_score: float | None = None,
) -> Tuple[float, dict]:
    """
    Compute overall similarity score between a new listing and a candidate.

    Returns:
        (combined_score, detail_dict) where combined_score is 0.0-1.0
        and detail_dict contains individual scores for debugging.
    """
    # Title similarity (fuzzy)
    new_title = new_listing.get("title", "")
    cand_title = candidate.get("title", "")
    title_score = fuzzy_ratio(new_title, cand_title)

    # Description similarity (TF-IDF, pre-computed if available)
    if desc_tfidf_score is not None:
        desc_score = desc_tfidf_score
    else:
        new_desc = new_listing.get("description", "")
        cand_desc = candidate.get("description", "")
        desc_score = tfidf_cosine_similarity(new_desc, cand_desc)

    # Address similarity
    new_addr = strip_diacritics(normalize_vietnamese(new_listing.get("address", "")))
    cand_addr = strip_diacritics(normalize_vietnamese(candidate.get("address", "")))
    addr_score = fuzzy_ratio(new_addr, cand_addr) if new_addr and cand_addr else 0.0

    # Price similarity (close price = more suspicious)
    new_price = new_listing.get("price", 0)
    cand_price = candidate.get("price", 0)
    if new_price and cand_price:
        price_ratio = min(new_price, cand_price) / max(new_price, cand_price)
        price_score = price_ratio if price_ratio > 0.8 else 0.0
    else:
        price_score = 0.0

    # Weighted combination
    combined = (
        title_score * 0.35 + desc_score * 0.35 + addr_score * 0.20 + price_score * 0.10
    )

    detail = {
        "titleSimilarity": round(title_score, 4),
        "descriptionSimilarity": round(desc_score, 4),
        "addressSimilarity": round(addr_score, 4),
        "priceSimilarity": round(price_score, 4),
        "combinedScore": round(combined, 4),
    }

    return combined, detail
