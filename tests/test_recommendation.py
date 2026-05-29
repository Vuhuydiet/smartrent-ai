"""
=================================================================
  Comprehensive Recommendation Test Suite (pytest / TestClient)
=================================================================

Covers the two recommendation endpoints end to end:
  - POST /api/v1/recommendations/similar
  - POST /api/v1/recommendations/personalized

Sections
  1. Basic contract           - response shape & ordering
  2. Geo accuracy             - haversine + distance-decay law
  3. /similar scenarios       - S1..S10 (distance, VIP, freshness, price, type, no-GPS)
  4. /personalized scenarios  - P1..P10 (location, CF, VIP, freshness, dedup, decay)
  5. Shift position-pinning    - recent location shift surfaces listings at pos 8/9/10

All listings without lat/lon fall back to administrative virtual distance
(ward > district > province), which is what the shift feature keys off.
"""
import math

import numpy as np

from fastapi.testclient import TestClient

from app.main import app
from app.service.recommendation_service import RecommendationService

client = TestClient(app)
BASE = "/api/v1/recommendations"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_listing(
    lid,
    product="ROOM",
    ltype="RENT",
    price=3_000_000,
    area=20.0,
    beds=1,
    province="01",
    district=1,
    ward_id=1,
    ward_code="00001",
    vip="NORMAL",
    days_ago=5,
    lat=None,
    lon=None,
) -> dict:
    return {
        "listing_id": lid,
        "product_type": product,
        "listing_type": ltype,
        "price": float(price),
        "area": float(area),
        "bedrooms": beds,
        "province_code": province,
        "district_id": district,
        "ward_id": ward_id,
        "ward_code": ward_code,
        "vip_type": vip,
        "post_date_days_ago": days_ago,
        "latitude": lat,
        "longitude": lon,
    }


def interaction(user, lid, weight) -> dict:
    return {"user_id": user, "listing_id": lid, "weight": weight}


def post_similar(payload) -> list:
    resp = client.post(f"{BASE}/similar", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def post_personalized(payload) -> list:
    resp = client.post(f"{BASE}/personalized", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


def haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


# Hồ Hoàn Kiếm, Hà Nội / Bến Thành, TP.HCM
HANOI = (21.0285, 105.8542)
HCMC = (10.7769, 106.7009)
KM_PER_DEG_LAT = 2.0 * math.pi * 6371.0 / 360.0

# Shared target + 15-candidate pool used by the S/P scenario tests.
TARGET = make_listing(
    9000,
    lat=21.0285,
    lon=105.8542,
    province="01",
    district=1,
    ward_id=1,
    ward_code="00001",
    days_ago=5,
)

POOL = [
    make_listing(
        1,
        lat=21.0315,
        lon=105.8542,
        price=3_200_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=3,
    ),
    make_listing(
        2,
        lat=21.0420,
        lon=105.8542,
        price=3_100_000,
        province="01",
        district=2,
        ward_id=2,
        ward_code="00002",
        days_ago=7,
    ),
    make_listing(
        3,
        lat=21.0735,
        lon=105.8542,
        price=2_900_000,
        province="01",
        district=5,
        ward_id=5,
        ward_code="00005",
        days_ago=10,
    ),
    make_listing(
        4,
        lat=21.1631,
        lon=105.8542,
        price=3_000_000,
        province="01",
        district=10,
        ward_id=10,
        ward_code="00010",
        days_ago=2,
    ),
    make_listing(
        5,
        lat=10.7769,
        lon=106.7009,
        price=3_000_000,
        province="79",
        district=100,
        ward_id=100,
        ward_code="99999",
        days_ago=1,
    ),
    make_listing(
        6,
        lat=None,
        lon=None,
        price=3_000_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=4,
    ),
    make_listing(
        7,
        lat=None,
        lon=None,
        price=3_000_000,
        province="48",
        district=200,
        ward_id=200,
        ward_code="88888",
        days_ago=4,
    ),
    make_listing(
        8,
        lat=21.0320,
        lon=105.8542,
        price=3_100_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=3,
        vip="DIAMOND",
    ),
    make_listing(
        9,
        lat=21.0310,
        lon=105.8542,
        price=3_050_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=6,
        vip="GOLD",
    ),
    make_listing(
        10,
        lat=21.0285,
        lon=105.8550,
        price=3_000_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=95,
    ),
    make_listing(
        11,
        product="APARTMENT",
        lat=21.0300,
        lon=105.8542,
        price=5_000_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=5,
    ),
    make_listing(
        12,
        ltype="SALE",
        lat=21.0295,
        lon=105.8542,
        price=3_000_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=5,
    ),
    make_listing(
        13,
        lat=21.0400,
        lon=105.8542,
        price=8_000_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=5,
    ),
    make_listing(
        14,
        lat=16.0544,
        lon=108.2022,
        price=3_000_000,
        province="48",
        district=300,
        ward_id=300,
        ward_code="77777",
        days_ago=5,
    ),
    make_listing(
        15,
        lat=21.0285,
        lon=105.8542,
        price=3_000_000,
        province="01",
        district=1,
        ward_id=1,
        ward_code="00001",
        days_ago=0,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BASIC CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════


def test_similar_listings():
    """Response is a sorted list of {listing_id, score, cf_score, cbf_score}."""
    payload = {
        "target": make_listing(
            100, product="APARTMENT", price=10_000_000.0, area=50.0, beds=2, days_ago=2
        ),
        "candidates": [
            make_listing(
                101,
                product="APARTMENT",
                price=11_000_000.0,
                area=55.0,
                beds=2,
                days_ago=3,
            ),
            make_listing(
                102,
                product="ROOM",
                price=3_000_000.0,
                area=20.0,
                beds=1,
                province="02",
                district=2,
                ward_id=2,
                ward_code="00002",
                days_ago=10,
            ),
        ],
        "top_n": 2,
    }
    data = post_similar(payload)
    assert isinstance(data, list)
    assert len(data) == 2
    assert {"listing_id", "score", "cf_score", "cbf_score"} <= data[0].keys()
    scores = [item["score"] for item in data]
    assert scores[0] >= scores[1]


def test_personalized_feed():
    """Personalized feed returns top_n items sorted by score descending."""
    payload = {
        "user_id": "test_user_789",
        "user_interactions": [
            interaction("test_user_789", 101, 3.0),
            interaction("test_user_789", 102, 1.0),
        ],
        "all_interactions": [
            interaction("test_user_789", 101, 3.0),
            interaction("other_user_1", 101, 2.5),
            interaction("other_user_1", 103, 3.0),
        ],
        "candidates": [
            make_listing(
                101,
                product="APARTMENT",
                price=8_000_000.0,
                area=45.0,
                beds=1,
                vip="SILVER",
                days_ago=1,
            ),
            make_listing(
                102,
                product="ROOM",
                price=3_000_000.0,
                area=20.0,
                beds=1,
                district=2,
                days_ago=5,
            ),
            make_listing(
                103,
                product="APARTMENT",
                price=8_500_000.0,
                area=48.0,
                beds=1,
                vip="GOLD",
                days_ago=0,
            ),
        ],
        "top_n": 3,
    }
    data = post_personalized(payload)
    assert isinstance(data, list)
    assert len(data) == 3
    assert "listing_id" in data[0] and "score" in data[0]
    scores = [item["score"] for item in data]
    assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. GEO ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════


def _geo_listing(listing_id, lat=None, lon=None, **overrides):
    base = {
        "listing_id": listing_id,
        "product_type": "APARTMENT",
        "listing_type": "RENT",
        "price": 10_000_000.0,
        "area": 50.0,
        "bedrooms": 2,
        "province_code": "01",
        "district_id": 1,
        "ward_id": 100,
        "ward_code": "00100",
        "vip_type": "NORMAL",
        "post_date_days_ago": 2,
    }
    base.update(overrides)
    if lat is not None:
        base["latitude"] = lat
    if lon is not None:
        base["longitude"] = lon
    return base


def test_geo_api_accepts_lat_lon():
    payload = {
        "target": _geo_listing(100, lat=HANOI[0], lon=HANOI[1]),
        "candidates": [
            _geo_listing(101, lat=HANOI[0] + 0.01, lon=HANOI[1]),
            _geo_listing(102, lat=HANOI[0] + 0.20, lon=HANOI[1]),
        ],
        "top_n": 2,
    }
    data = post_similar(payload)
    assert len(data) == 2
    for item in data:
        assert {"listing_id", "score", "cf_score", "cbf_score"} <= item.keys()
        assert item["score"] >= 0.0


def test_geo_haversine_numeric_accuracy():
    svc = RecommendationService()
    d_service = svc._calculate_haversine_vectorized(
        HANOI[0], HANOI[1], np.array([HCMC[0]]), np.array([HCMC[1]])
    )[0]
    d_ref = haversine(*HANOI, *HCMC)
    assert abs(d_service - d_ref) < 1e-6, (d_service, d_ref)
    assert 1100.0 < d_service < 1200.0, f"Hanoi-HCMC = {d_service:.1f} km out of range"

    for deg in (0.018, 0.09, 0.45):
        d = svc._calculate_haversine_vectorized(
            HANOI[0], HANOI[1], np.array([HANOI[0] + deg]), np.array([HANOI[1]])
        )[0]
        assert abs(d - deg * KM_PER_DEG_LAT) < 0.05, (deg, d)


def test_geo_similar_ranking_follows_distance():
    """Identical content → ranking is pure geo decay: score(d)/score(0) = exp(-k·d)."""
    dists_km = {101: 0.0, 102: 2.0, 103: 10.0, 104: 50.0}
    candidates = [
        _geo_listing(lid, lat=HANOI[0] + km / KM_PER_DEG_LAT, lon=HANOI[1])
        for lid, km in dists_km.items()
    ]
    data = post_similar(
        {
            "target": _geo_listing(100, lat=HANOI[0], lon=HANOI[1]),
            "candidates": candidates,
            "top_n": 4,
        }
    )

    assert [item["listing_id"] for item in data] == [101, 102, 103, 104]
    for item in data:
        assert abs(item["cbf_score"] - 1.0) < 1e-3, item
    scores = [item["score"] for item in data]
    assert all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)), scores

    decay_k = RecommendationService().DECAY_COEFFICIENT
    by_id = {item["listing_id"]: item["score"] for item in data}
    s0 = by_id[101]
    for lid, km in dists_km.items():
        assert abs(by_id[lid] / s0 - math.exp(-decay_k * km)) < 1e-3, (lid, km)


def test_geo_latlon_overrides_virtual_distance():
    # No GPS: identical content + identical virtual distance → equal scores.
    no_gps = post_similar(
        {
            "target": _geo_listing(200),
            "candidates": [_geo_listing(201), _geo_listing(202)],
            "top_n": 2,
        }
    )
    s = {i["listing_id"]: i["score"] for i in no_gps}
    assert abs(s[201] - s[202]) < 1e-6, s

    # With GPS: 201 on target, 202 ~50 km away → 201 wins clearly.
    data = post_similar(
        {
            "target": _geo_listing(200, lat=HANOI[0], lon=HANOI[1]),
            "candidates": [
                _geo_listing(201, lat=HANOI[0], lon=HANOI[1]),
                _geo_listing(202, lat=HANOI[0] + 50.0 / KM_PER_DEG_LAT, lon=HANOI[1]),
            ],
            "top_n": 2,
        }
    )
    assert data[0]["listing_id"] == 201, data
    g = {i["listing_id"]: i["score"] for i in data}
    assert g[201] > g[202] * 5, g


def test_geo_personalized_uses_lat_lon():
    user = "geo_user"
    data = post_personalized(
        {
            "user_id": user,
            "user_interactions": [interaction(user, 301, 3.0)],
            "all_interactions": [],
            "interaction_features": [_geo_listing(301, lat=HANOI[0], lon=HANOI[1])],
            "candidates": [
                _geo_listing(401, lat=HANOI[0] + 1.0 / KM_PER_DEG_LAT, lon=HANOI[1]),
                _geo_listing(402, lat=HANOI[0] + 40.0 / KM_PER_DEG_LAT, lon=HANOI[1]),
            ],
            "top_n": 2,
        }
    )
    assert [i["listing_id"] for i in data] == [401, 402]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. /similar SCENARIOS  (S1..S10)
# ═══════════════════════════════════════════════════════════════════════════════


def test_similar_s1_distance_ordering():
    candidates = [
        make_listing(1, lat=21.0315, lon=105.8542, days_ago=5),
        make_listing(2, lat=21.0420, lon=105.8542, days_ago=5),
        make_listing(3, lat=21.0735, lon=105.8542, days_ago=5),
        make_listing(4, lat=21.1631, lon=105.8542, days_ago=5),
        make_listing(5, lat=10.7769, lon=106.7009, days_ago=5, province="79"),
    ]
    data = post_similar({"target": TARGET, "candidates": candidates, "top_n": 10})
    assert [r["listing_id"] for r in data] == [1, 2, 3, 4, 5]


def test_similar_s2_vip_boost():
    candidates = [POOL[i - 1] for i in [1, 8, 9, 2, 3]]
    data = post_similar({"target": TARGET, "candidates": candidates, "top_n": 10})
    score = {r["listing_id"]: r["score"] for r in data}
    assert data[0]["listing_id"] == 9  # GOLD, closest → top
    assert score[8] > score[1]  # DIAMOND beats NORMAL despite farther


def test_similar_s3_freshness_decay():
    candidates = [POOL[i - 1] for i in [10, 1, 2]]
    data = post_similar({"target": TARGET, "candidates": candidates, "top_n": 10})
    assert {r["listing_id"] for r in data} == {10, 1, 2}
    assert all(r["score"] >= 0.0 for r in data)


def test_similar_s4_virtual_distance():
    candidates = [POOL[i - 1] for i in [6, 7]]  # same-district vs Da Nang (no GPS)
    data = post_similar({"target": TARGET, "candidates": candidates, "top_n": 10})
    score = {r["listing_id"]: r["score"] for r in data}
    assert score[6] > score[7]


def test_similar_s5_product_type_mismatch():
    candidates = [POOL[i - 1] for i in [11, 1, 2]]  # APARTMENT vs ROOM
    data = post_similar({"target": TARGET, "candidates": candidates, "top_n": 10})
    score = {r["listing_id"]: r["score"] for r in data}
    assert score[1] > score[11]


def test_similar_s6_price_mismatch():
    candidates = [POOL[i - 1] for i in [13, 2, 1]]  # 8M vs 3.1M vs 3.2M
    data = post_similar({"target": TARGET, "candidates": candidates, "top_n": 10})
    score = {r["listing_id"]: r["score"] for r in data}
    assert score[2] > score[13]


def test_similar_s7_with_personalization():
    candidates = [POOL[i - 1] for i in [1, 2, 3, 4, 5, 6]]
    data = post_similar(
        {
            "target": TARGET,
            "candidates": candidates,
            "top_n": 10,
            "user_interactions": [interaction("u1", 2, 3.0), interaction("u1", 3, 1.0)],
            "interaction_features": [POOL[1], POOL[2]],
        }
    )
    assert len(data) == 6
    assert any(r["cf_score"] > 0 for r in data)  # personalization signal present


def test_similar_s8_no_gps_target():
    target_no_gps = make_listing(8888, days_ago=3)  # no lat/lon
    candidates = [POOL[i - 1] for i in [6, 7, 1, 2]]
    data = post_similar(
        {"target": target_no_gps, "candidates": candidates, "top_n": 10}
    )
    assert len(data) == 4
    assert data[0]["listing_id"] == 6  # same ward wins via virtual distance


def test_similar_s9_all_candidates_no_gps():
    candidates = [
        make_listing(101, province="01", district=1, ward_id=1, ward_code="00001"),
        make_listing(102, province="01", district=2, ward_id=2, ward_code="00002"),
        make_listing(103, province="48", district=200, ward_id=200, ward_code="88888"),
    ]
    data = post_similar({"target": TARGET, "candidates": candidates, "top_n": 10})
    assert data[0]["listing_id"] == 101  # same ward first


def test_similar_s10_full_pool():
    data = post_similar({"target": TARGET, "candidates": POOL, "top_n": 10})
    ids = [r["listing_id"] for r in data]
    scores = [r["score"] for r in data]
    assert len(ids) == len(set(ids))  # no dup
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))  # sorted
    assert 5 not in ids[:3]  # HCMC not top-3


# ═══════════════════════════════════════════════════════════════════════════════
# 4. /personalized SCENARIOS  (P1..P10)
# ═══════════════════════════════════════════════════════════════════════════════


def test_personalized_p1_strong_location():
    candidates = [POOL[i - 1] for i in [1, 2, 3, 4, 5, 6, 7]]
    data = post_personalized(
        {
            "user_id": "user-p1",
            "user_interactions": [
                interaction("user-p1", 9001, 3.0),
                interaction("user-p1", 9002, 2.5),
                interaction("user-p1", 9003, 1.0),
            ],
            "all_interactions": [
                interaction("user-p1", 9001, 3.0),
                interaction("user-p2", 1, 2.5),
                interaction("user-p2", 2, 1.0),
            ],
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [
                make_listing(9001, lat=21.0285, lon=105.8545),
                make_listing(9002, lat=21.0290, lon=105.8540),
                make_listing(9003, lat=21.0280, lon=105.8538),
            ],
        }
    )
    top3 = [r["listing_id"] for r in data[:3]]
    assert any(i in {1, 6} for i in top3)  # Hoan Kiem listings dominate


def test_personalized_p2_cf_signal():
    candidates = [POOL[i - 1] for i in [1, 2, 3, 5, 14]]
    all_interactions = (
        [interaction("user-p2", 9001, 1.0)]
        + [interaction(f"u{i}", 9001, 1.0) for i in range(10, 15)]
        + [interaction(f"u{i}", 2, 1.0) for i in range(10, 15)]  # co-view of id=2
    )
    data = post_personalized(
        {
            "user_id": "user-p2",
            "user_interactions": [interaction("user-p2", 9001, 1.0)],
            "all_interactions": all_interactions,
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [make_listing(9001, lat=21.0285, lon=105.8542)],
        }
    )
    cf = {r["listing_id"]: r["cf_score"] for r in data}
    assert cf.get(2, 0) > 0


def test_personalized_p3_vip_freshness_combo():
    candidates = [POOL[i - 1] for i in [8, 9, 1, 10]]
    data = post_personalized(
        {
            "user_id": "user-p3",
            "user_interactions": [interaction("user-p3", 9001, 2.0)],
            "all_interactions": [interaction("user-p3", 9001, 2.0)],
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [make_listing(9001, lat=21.0285, lon=105.8542)],
        }
    )
    score = {r["listing_id"]: r["score"] for r in data}
    assert score[10] < score[9]  # 95-day stale ranks below fresh GOLD


def test_personalized_p4_no_interaction_features():
    candidates = [POOL[i - 1] for i in [1, 2, 3, 5, 6, 7]]
    data = post_personalized(
        {
            "user_id": "user-p4",
            "user_interactions": [interaction("user-p4", 9001, 1.0)],
            "all_interactions": [interaction("user-p4", 9001, 1.0)],
            "candidates": candidates,
            "top_n": 10,
        }
    )
    assert len(data) == 6  # CBF-only fallback still returns a feed


def test_personalized_p5_geo_spread():
    candidates = [POOL[i - 1] for i in [1, 2, 3, 6, 7, 14]]
    data = post_personalized(
        {
            "user_id": "user-p5",
            "user_interactions": [
                interaction("user-p5", 9001, 3.0),
                interaction("user-p5", 9002, 1.0),
                interaction("user-p5", 9003, 1.0),
            ],
            "all_interactions": [
                interaction("user-p5", 9001, 3.0),
                interaction("user-p5", 9002, 1.0),
                interaction("user-p5", 9003, 1.0),
            ],
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [
                make_listing(9001, lat=21.0285, lon=105.8542, days_ago=30),
                make_listing(9002, lat=21.0290, lon=105.8540, days_ago=15),
                make_listing(
                    9003,
                    lat=16.0544,
                    lon=108.2022,
                    province="48",
                    district=300,
                    ward_id=300,
                    ward_code="77777",
                    days_ago=1,
                ),
            ],
        }
    )
    assert len(data) == 6


def test_personalized_p6_vip_tier_ordering():
    candidates = [
        make_listing(201, vip="DIAMOND", lat=21.0300, lon=105.8542),
        make_listing(202, vip="GOLD", lat=21.0300, lon=105.8542),
        make_listing(203, vip="SILVER", lat=21.0300, lon=105.8542),
        make_listing(204, vip="NORMAL", lat=21.0300, lon=105.8542),
    ]
    data = post_personalized(
        {
            "user_id": "user-p6",
            "user_interactions": [interaction("user-p6", 9001, 1.0)],
            "all_interactions": [interaction("user-p6", 9001, 1.0)],
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [make_listing(9001, lat=21.0285, lon=105.8542)],
        }
    )
    assert [r["listing_id"] for r in data] == [201, 202, 203, 204]


def test_personalized_p7_freshness_extremes():
    candidates = [
        make_listing(301, lat=21.0285, lon=105.8542, days_ago=0),
        make_listing(302, lat=21.0285, lon=105.8542, days_ago=50),
        make_listing(303, lat=21.0285, lon=105.8542, days_ago=100),
    ]
    data = post_personalized(
        {
            "user_id": "user-p7",
            "user_interactions": [interaction("user-p7", 9001, 1.0)],
            "all_interactions": [interaction("user-p7", 9001, 1.0)],
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [make_listing(9001, lat=21.0285, lon=105.8542)],
        }
    )
    assert data[0]["listing_id"] == 301  # freshest first


def test_personalized_p8_duplicate_dedup():
    candidates = [POOL[0], POOL[0], POOL[1], POOL[2]]  # id=1 twice
    data = post_personalized(
        {
            "user_id": "user-p8",
            "user_interactions": [interaction("user-p8", 9001, 1.0)],
            "all_interactions": [interaction("user-p8", 9001, 1.0)],
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [make_listing(9001, lat=21.0285, lon=105.8542)],
        }
    )
    ids = [r["listing_id"] for r in data]
    assert len(ids) == len(set(ids))


def test_personalized_p9_long_distance_decay():
    candidates = [POOL[i - 1] for i in [1, 2, 5, 14]]
    data = post_personalized(
        {
            "user_id": "user-p9",
            "user_interactions": [interaction("user-p9", 9001, 1.0)],
            "all_interactions": [interaction("user-p9", 9001, 1.0)],
            "candidates": candidates,
            "top_n": 10,
            "interaction_features": [make_listing(9001, lat=21.0285, lon=105.8542)],
        }
    )
    score = {r["listing_id"]: r["score"] for r in data}
    assert score[1] > 0
    assert score[1] > score.get(5, 0)  # nearby beats HCMC


def test_personalized_p10_full_pool():
    data = post_personalized(
        {
            "user_id": "user-p10",
            "user_interactions": [
                interaction("user-p10", 9001, 3.0),
                interaction("user-p10", 9002, 2.5),
                interaction("user-p10", 9003, 1.0),
            ],
            "all_interactions": [
                interaction("user-p10", 9001, 3.0),
                interaction("user-p10", 9002, 2.5),
                interaction("user-p10", 9003, 1.0),
                interaction("user-p11", 1, 2.0),
                interaction("user-p11", 2, 1.0),
                interaction("user-p12", 8, 3.0),
            ],
            "candidates": POOL,
            "top_n": 10,
            "interaction_features": [
                make_listing(9001, lat=21.0285, lon=105.8542, days_ago=5),
                make_listing(9002, lat=21.0290, lon=105.8540, days_ago=3),
                make_listing(9003, lat=21.0280, lon=105.8545, days_ago=1),
            ],
        }
    )
    ids = [r["listing_id"] for r in data]
    scores = [r["score"] for r in data]
    assert len(ids) == len(set(ids))
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert 5 not in ids[:5] and 14 not in ids[:5]  # far listings not in top-5


def test_personalized_p11_meets_shift_condition_false_preserves_preferred_location():
    # User preferred location is Hanoi (province "01", district 1) since they have more history there.
    # But they just viewed one listing in HCMC (province "79", district 100).
    user = "user-p11"

    # Kịch bản 1: meets_shift_condition = False
    # Tâm tham chiếu địa lý phải là Hà Nội (Listing 9001). Tin Hà Nội (id=1) phải thắng tin HCMC (id=5).
    data_no_shift = post_personalized(
        {
            "user_id": user,
            "user_interactions": [
                interaction(user, 9001, 3.0),  # Hanoi
                interaction(user, 9003, 1.0),  # Hanoi
                interaction(user, 9002, 1.0),  # HCMC (most recent)
            ],
            "all_interactions": [],
            "candidates": [POOL[0], POOL[4]],  # id=1 (Hanoi), id=5 (HCMC)
            "top_n": 2,
            "interaction_features": [
                make_listing(
                    9002,
                    lat=HCMC[0],
                    lon=HCMC[1],
                    province="79",
                    district=100,
                    ward_id=100,
                    ward_code="99999",
                ),  # HCMC (most recent)
                make_listing(
                    9001, lat=HANOI[0], lon=HANOI[1], province="01", district=1
                ),  # Hanoi (preferred)
                make_listing(
                    9003,
                    lat=HANOI[0] + 0.001,
                    lon=HANOI[1] + 0.001,
                    province="01",
                    district=1,
                ),  # Hanoi
            ],
            "meets_shift_condition": False,
        }
    )
    assert data_no_shift[0]["listing_id"] == 1  # Hanoi listing wins!

    # Kịch bản 2: meets_shift_condition = True (BLEND geo-anchor)
    # Geo-decay được neo vào CẢ HAI: preferred (Hà Nội) + discovery (HCMC), lấy
    # khoảng cách nhỏ hơn. Nhờ vậy tin HCMC (id=5) KHÔNG còn bị triệt tiêu như khi
    # no-shift; nhưng tin preferred (Hà Nội id=1) cũng không bị triệt tiêu và vẫn
    # dẫn đầu xếp hạng relevance của Python (profile nghiêng Hà Nội). Việc surface
    # tin discovery lên slot 8/9/10 do backend PIN, không phải Python lật cả feed.
    data_shift = post_personalized(
        {
            "user_id": user,
            "user_interactions": [
                interaction(user, 9001, 3.0),  # Hanoi
                interaction(user, 9003, 1.0),  # Hanoi
                interaction(user, 9002, 1.0),  # HCMC (most recent)
            ],
            "all_interactions": [],
            "candidates": [POOL[0], POOL[4]],  # id=1 (Hanoi), id=5 (HCMC)
            "top_n": 2,
            "interaction_features": [
                make_listing(
                    9002,
                    lat=HCMC[0],
                    lon=HCMC[1],
                    province="79",
                    district=100,
                    ward_id=100,
                    ward_code="99999",
                ),  # HCMC (most recent)
                make_listing(
                    9001, lat=HANOI[0], lon=HANOI[1], province="01", district=1
                ),  # Hanoi (preferred)
                make_listing(
                    9003,
                    lat=HANOI[0] + 0.001,
                    lon=HANOI[1] + 0.001,
                    province="01",
                    district=1,
                ),  # Hanoi
            ],
            "meets_shift_condition": True,
        }
    )

    def _score(data, lid):
        return next(r["score"] for r in data if r["listing_id"] == lid)

    # Blend "giải cứu" tin discovery: điểm id=5 khi shift cao hơn hẳn khi no-shift
    # (no-shift nó bị decay ~1000km về gần 0).
    assert _score(data_shift, 5) > _score(data_no_shift, 5)
    # Preferred vẫn dẫn đầu — feed KHÔNG bị lật toàn bộ sang nơi mới.
    assert data_shift[0]["listing_id"] == 1
