from app.agent.suggestions import build_suggestions


def _listings(n):
    return [{"listingId": 100 + i, "title": f"Tin {i}"} for i in range(n)]


def test_search_with_results_offers_detail_compare_next():
    out = build_suggestions(["search_listings"], _listings(3), has_auth=True)
    labels = [s["label"] for s in out]
    assert "Xem chi tiết căn 1" in labels
    assert any("So sánh" in lbl for lbl in labels)
    assert any("Xem tiếp" in lbl for lbl in labels)
    assert 1 <= len(out) <= 4


def test_search_single_result_has_no_compare():
    out = build_suggestions(["search_listings"], _listings(1), has_auth=True)
    assert not any("So sánh" in s["label"] for s in out)


def test_no_chip_query_exposes_a_listing_id():
    """The user must never see or send a listing ID — chips are positional."""
    contexts = [
        (["search_listings"], _listings(3)),
        (["get_recommendations"], _listings(3)),
        (["get_listing_detail"], _listings(1)),
        (["compare_listings"], _listings(3)),
        (["my_listings_status"], []),
        ([], []),
    ]
    for tools_used, listings in contexts:
        for has_auth in (True, False):
            for chip in build_suggestions(tools_used, listings, has_auth):
                blob = f"{chip['label']} {chip['query']}"
                assert "Mã tin" not in blob
                assert "#" not in blob
                # _listings() ids are 100, 101, 102 — none may appear.
                assert not any(str(100 + i) in blob for i in range(3))


def test_compare_chip_covers_the_whole_result_set():
    """Comparison is one operation: all results, never a chosen pair."""
    out = build_suggestions(["search_listings"], _listings(4), has_auth=True)
    compare = next(s for s in out if "So sánh" in s["label"])
    assert compare["label"] == "So sánh tất cả"
    assert "tất cả" in compare["query"]
    # No chip may ask for a subset comparison.
    assert not any("2 căn" in s["label"] for s in out)


def test_search_zero_results_offers_relaxation():
    out = build_suggestions(["search_listings"], [], has_auth=True)
    labels = [s["label"] for s in out]
    assert "Tăng ngân sách" in labels
    assert len(out) >= 1


def test_detail_context_offers_compare_save_nearby():
    out = build_suggestions(["get_listing_detail"], _listings(1), has_auth=True)
    labels = [s["label"] for s in out]
    assert "Lưu tin này" in labels
    assert any("Xung quanh" in lbl for lbl in labels)


def test_owner_context():
    out = build_suggestions(["my_listings_status"], [], has_auth=True)
    assert any("hết hạn" in s["label"].lower() for s in out)


def test_detail_guest_hides_save_chip():
    out = build_suggestions(["get_listing_detail"], _listings(1), has_auth=False)
    labels = [s["label"] for s in out]
    assert "Lưu tin này" not in labels
    assert any("Xung quanh" in lbl for lbl in labels)


def test_owner_context_empty_for_guest():
    assert build_suggestions(["my_listings_status"], [], has_auth=False) == []


def test_starter_when_no_tool_and_no_listings():
    out = build_suggestions([], [], has_auth=False)
    assert len(out) == 3
    assert all(s["query"] for s in out)


def test_unknown_context_returns_empty():
    out = build_suggestions(["get_user_info"], [], has_auth=True)
    assert out == []


def test_cap_four():
    out = build_suggestions(["search_listings"], _listings(5), has_auth=True)
    assert len(out) <= 4
