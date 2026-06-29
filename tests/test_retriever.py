"""Tests for the RAG retriever's text normalisation and location codes."""

from app.agent.rag.retriever import RAGRetriever, _keyword_score, _normalise


def test_normalise_folds_d_stroke_to_plain_d():
    # đ/Đ (U+0111/U+0110) are base letters, NOT combining accents, so the NFKD
    # strip leaves them intact. They must fold to "d", otherwise no-diacritic
    # queries ("dat coc", "dang tin", "dieu hoa") — a very common typing style —
    # silently miss every KB entry that canonically uses đ ("đặt cọc",
    # "đăng tin", "điều hòa").
    assert _normalise("Đặt Cọc") == "dat coc"
    assert _normalise("đăng tin") == "dang tin"
    assert _normalise("điều hòa") == "dieu hoa"


def test_location_context_emits_district_code_string():
    # The retriever must hand the model districtCode (GSO string), not a numeric
    # districtId, so the backend can resolve it to the surrogate PK.
    ctx = RAGRetriever()._location_context(_normalise("tìm trọ quận 1"))
    assert 'districtCode="760"' in ctx
    assert "districtId=" not in ctx


def test_keyword_score_short_keyword_requires_whole_word():
    # Short keywords (<=3 chars) matched as raw substrings produced noisy
    # off-topic FAQ/guide suggestions (e.g. "an" inside "ngan hang"). They must
    # only count as a standalone word now.
    assert _keyword_score("ngan hang", ["an"]) == 0
    # …but still match as a real word, and long keywords keep substring behaviour.
    assert _keyword_score("goi vip cua toi", ["vip"]) == 1
    assert _keyword_score("tien dat coc", ["dat coc"]) == 1


def test_location_context_emits_province_without_kb_districts():
    # A province with no districts in the KB (e.g. Cần Thơ = 92) must still emit
    # its provinceCode. The old province-only branch lived inside the district
    # loop, so province-less provinces emitted nothing at all.
    ctx = RAGRetriever()._location_context(_normalise("tìm phòng ở Cần Thơ"))
    assert 'provinceCode="92"' in ctx
