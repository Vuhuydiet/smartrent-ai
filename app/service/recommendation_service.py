from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from app.dto.recommendation import (
    InteractionEntry,
    ListingFeature,
    PersonalizedFeedRequest,
    RecommendationItem,
    SimilarListingRequest,
)


class RecommendationService:
    def __init__(self):
        self.VIP_WEIGHTS = {
            "NORMAL": 1.0,
            "SILVER": 1.05,
            "GOLD": 1.10,
            "DIAMOND": 1.15,
        }
        self.VIP_RANKS = {"DIAMOND": 4, "GOLD": 3, "SILVER": 2, "NORMAL": 1}
        # Geospatial decay in exp(-k * dist_km). k = ln(2)/10 → score halves
        # every 10 km, so cross-district listings stay competitive.
        self.DECAY_COEFFICIENT = float(np.log(2.0) / 10.0)
        # Freshness boost weight: a brand-new listing gets +30%, a 100-day-old
        # one gets +0%. Larger weight = freshness matters more in ranking.
        self.FRESHNESS_WEIGHT = 0.3
        # MinMax normalization floor: the weakest candidate is mapped to this
        # value instead of 0.0, so a relevant listing never shows a 0 score.
        self.NORMALIZATION_FLOOR = 0.1

    def _calculate_haversine_vectorized(
        self,
        target_lat: float,
        target_lon: float,
        c_lats: np.ndarray,
        c_lons: np.ndarray,
    ) -> np.ndarray:
        t_lat_rad = np.radians(target_lat)
        t_lon_rad = np.radians(target_lon)
        c_lats_rad = np.radians(c_lats)
        c_lons_rad = np.radians(c_lons)

        dlat = c_lats_rad - t_lat_rad
        dlon = c_lons_rad - t_lon_rad

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(t_lat_rad) * np.cos(c_lats_rad) * np.sin(dlon / 2.0) ** 2
        )
        c = 2.0 * np.arcsin(np.sqrt(a))
        return c * 6371.0

    def _compute_virtual_distance(self, target, candidate) -> float:
        if (
            target.ward_id and candidate.ward_id and target.ward_id == candidate.ward_id
        ) or (
            target.ward_code
            and candidate.ward_code
            and target.ward_code == candidate.ward_code
        ):
            return 1.0
        if (
            target.district_id
            and candidate.district_id
            and target.district_id == candidate.district_id
        ):
            return 3.0
        if (
            target.province_code
            and candidate.province_code
            and target.province_code == candidate.province_code
        ):
            return 8.0
        return 35.0

    def _normalize_scores_minmax(self, scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val == min_val:
            return np.ones_like(scores)
        norm = (scores - min_val) / (max_val - min_val)
        floor = self.NORMALIZATION_FLOOR
        return floor + (1.0 - floor) * norm

    async def get_similar_listings(
        self, req: SimilarListingRequest
    ) -> List[RecommendationItem]:
        if not req.candidates:
            return []

        # Deduplicate candidates by listing_id
        seen = set()
        unique_candidates = []
        for c in req.candidates:
            if c.listing_id not in seen:
                seen.add(c.listing_id)
                unique_candidates.append(c)
        req.candidates = unique_candidates

        # Feature matrix — includes target, candidates, and historical interaction features
        all_listings_for_matrix = [req.target] + req.candidates
        if req.interaction_features:
            all_listings_for_matrix += req.interaction_features

        features_matrix, id_to_index = self._build_feature_matrix(
            all_listings_for_matrix
        )
        target_vec = features_matrix[id_to_index[req.target.listing_id]]

        # --- Build user profile vector (for personalization) ---
        has_personalization = (
            req.user_interactions is not None and len(req.user_interactions) > 0
        )
        profile_vector = np.zeros(features_matrix.shape[1])
        if has_personalization and req.user_interactions is not None:
            total_weight = 0.0
            for interaction in req.user_interactions:
                if interaction.listing_id in id_to_index:
                    w = interaction.weight
                    profile_vector += (
                        w * features_matrix[id_to_index[interaction.listing_id]]
                    )
                    total_weight += w
            if total_weight > 0:
                profile_vector /= total_weight
            else:
                has_personalization = False

        # Vectorized similarity and personalization computations (eliminates loop overhead)
        candidate_matrix = np.array(
            [features_matrix[id_to_index[c.listing_id]] for c in req.candidates]
        )
        raw_sim_scores = cosine_similarity(target_vec.reshape(1, -1), candidate_matrix)[
            0
        ]

        if has_personalization:
            raw_pers_scores = cosine_similarity(
                profile_vector.reshape(1, -1), candidate_matrix
            )[0]
        else:
            raw_pers_scores = np.zeros(len(req.candidates))

        # --- Stage 2: Calculate Base AI Score & MinMax Normalize to [0, 1] ---
        if has_personalization:
            base_scores = 0.9 * raw_sim_scores + 0.1 * raw_pers_scores
        else:
            base_scores = raw_sim_scores

        base_scores_norm = self._normalize_scores_minmax(base_scores)

        # --- Compute Physical or Virtual Distances ---
        distances = []
        target_lat = req.target.latitude
        target_lon = req.target.longitude

        has_gps_list = []
        c_lats_list = []
        c_lons_list = []
        for candidate in req.candidates:
            if (
                target_lat is not None
                and target_lon is not None
                and candidate.latitude is not None
                and candidate.longitude is not None
            ):
                has_gps_list.append(True)
                c_lats_list.append(candidate.latitude)
                c_lons_list.append(candidate.longitude)
            else:
                has_gps_list.append(False)
                c_lats_list.append(0.0)
                c_lons_list.append(0.0)

        has_gps = np.array(has_gps_list)
        c_lats = np.array(c_lats_list)
        c_lons = np.array(c_lons_list)

        if target_lat is not None and target_lon is not None and np.any(has_gps):
            gps_distances = self._calculate_haversine_vectorized(
                target_lat, target_lon, c_lats, c_lons
            )
        else:
            gps_distances = np.zeros(len(req.candidates))

        for idx, candidate in enumerate(req.candidates):
            if has_gps[idx]:
                distances.append(float(gps_distances[idx]))
            else:
                distances.append(self._compute_virtual_distance(req.target, candidate))

        dist_array = np.array(distances)
        decay_factors = np.exp(-self.DECAY_COEFFICIENT * dist_array)
        geospatial_scores = base_scores_norm * decay_factors

        # --- Score each candidate and apply VIP/Freshness Boosts ---
        results = []
        for idx, candidate in enumerate(req.candidates):
            geo_score = float(geospatial_scores[idx])
            pers_score = float(raw_pers_scores[idx])
            sim_score = float(raw_sim_scores[idx])

            vip_boost = self.VIP_WEIGHTS.get(candidate.vip_type.upper(), 1.0)
            freshness_boost = max(0.0, 1.0 - (candidate.post_date_days_ago * 0.01))
            final_score = (
                geo_score * vip_boost * (1.0 + self.FRESHNESS_WEIGHT * freshness_boost)
            )

            results.append(
                RecommendationItem(
                    listing_id=candidate.listing_id,
                    score=round(final_score, 4),
                    cf_score=round(pers_score, 4),
                    cbf_score=round(sim_score, 4),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: req.top_n]

    async def get_personalized_feed(
        self, req: PersonalizedFeedRequest
    ) -> List[RecommendationItem]:
        if not req.candidates:
            return []

        # Deduplicate candidates by listing_id
        seen = set()
        unique_candidates = []
        for c in req.candidates:
            if c.listing_id not in seen:
                seen.add(c.listing_id)
                unique_candidates.append(c)
        req.candidates = unique_candidates

        # Feature matrix
        all_listings_for_matrix = req.candidates
        if req.interaction_features:
            all_listings_for_matrix = req.interaction_features + req.candidates

        features_matrix, id_to_index = self._build_feature_matrix(
            all_listings_for_matrix
        )

        # Build Profile
        profile_vec, total_w = self._build_user_profile(
            req.user_interactions, features_matrix, id_to_index
        )

        # Preferred Location
        (
            pref_prov,
            pref_dist,
            pref_ward_id,
            pref_ward_code,
        ) = self._detect_preferred_location(req.interaction_features)

        # CF Computation
        cf_scores = self._compute_cf_scores(
            req.user_interactions or [],
            req.all_interactions or [],
            [c.listing_id for c in req.candidates],
        )

        # Vectorized CBF similarity computation (eliminates loop overhead)
        candidate_matrix = np.array(
            [features_matrix[id_to_index[c.listing_id]] for c in req.candidates]
        )
        if total_w > 0:
            cbf_scores_raw = cosine_similarity(
                profile_vec.reshape(1, -1), candidate_matrix
            )[0]
        else:
            cbf_scores_raw = np.zeros(len(req.candidates))

        # --- Calculate Base Hybrid Scores for candidates ---
        base_scores_list = []
        for idx, candidate in enumerate(req.candidates):
            cf_val = cf_scores.get(candidate.listing_id, 0.0)
            cbf_val = float(cbf_scores_raw[idx])
            if cf_val > 0:
                bs = (0.4 * cf_val) + (0.6 * cbf_val)
            else:
                bs = cbf_val * 0.9
            base_scores_list.append(bs)
        base_scores = np.array(base_scores_list)

        # MinMax Normalization with safety guard clause
        base_scores_norm = self._normalize_scores_minmax(base_scores)

        # --- Compute Physical or Virtual Distances ---
        target_lat = None
        target_lon = None
        ref_listing = None
        if req.interaction_features:
            for feat in req.interaction_features:
                if feat.latitude is not None and feat.longitude is not None:
                    target_lat = feat.latitude
                    target_lon = feat.longitude
                    ref_listing = feat
                    break

        distances = []
        has_gps_list = []
        c_lats_list = []
        c_lons_list = []
        for candidate in req.candidates:
            if (
                target_lat is not None
                and target_lon is not None
                and candidate.latitude is not None
                and candidate.longitude is not None
            ):
                has_gps_list.append(True)
                c_lats_list.append(candidate.latitude)
                c_lons_list.append(candidate.longitude)
            else:
                has_gps_list.append(False)
                c_lats_list.append(0.0)
                c_lons_list.append(0.0)

        has_gps = np.array(has_gps_list)
        c_lats = np.array(c_lats_list)
        c_lons = np.array(c_lons_list)

        if target_lat is not None and target_lon is not None and np.any(has_gps):
            gps_distances = self._calculate_haversine_vectorized(
                target_lat, target_lon, c_lats, c_lons
            )
        else:
            gps_distances = np.zeros(len(req.candidates))

        # Fallback VirtualTarget class for unified administrative matching
        class VirtualTarget:
            def __init__(self, province_code, district_id, ward_id, ward_code):
                self.province_code = province_code
                self.district_id = district_id
                self.ward_id = ward_id
                self.ward_code = ward_code

        fallback_target: Any = ref_listing
        if fallback_target is None:
            fallback_target = VirtualTarget(
                pref_prov, pref_dist, pref_ward_id, pref_ward_code
            )

        for idx, candidate in enumerate(req.candidates):
            if has_gps[idx]:
                distances.append(float(gps_distances[idx]))
            else:
                distances.append(
                    self._compute_virtual_distance(fallback_target, candidate)
                )

        dist_array = np.array(distances)
        decay_factors = np.exp(-self.DECAY_COEFFICIENT * dist_array)
        geospatial_scores = base_scores_norm * decay_factors

        # --- Score each candidate and apply VIP/Freshness Boosts ---
        results = []
        for idx, candidate in enumerate(req.candidates):
            geo_score = float(geospatial_scores[idx])
            cf_val = cf_scores.get(candidate.listing_id, 0.0)
            cbf_val = float(cbf_scores_raw[idx])

            vip_boost = self.VIP_WEIGHTS.get(candidate.vip_type.upper(), 1.0)
            freshness_boost = max(0.0, 1.0 - (candidate.post_date_days_ago * 0.01))
            final_score = (
                geo_score * vip_boost * (1.0 + self.FRESHNESS_WEIGHT * freshness_boost)
            )

            results.append(
                RecommendationItem(
                    listing_id=candidate.listing_id,
                    score=round(final_score, 4),
                    cf_score=round(cf_val, 4),
                    cbf_score=round(cbf_val, 4),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        # Position pinning (discovery-shift slots 8/9/10) is handled by the Java
        # backend, which owns the final feed assembly. Here we only return the
        # relevance-ranked list.
        return results[: req.top_n]

    def _build_user_profile(
        self, interactions, features_matrix, id_to_index
    ) -> Tuple[np.ndarray, float]:
        profile_vector = np.zeros(features_matrix.shape[1])
        total_weight = 0.0
        if interactions:
            for interaction in interactions:
                if interaction.listing_id in id_to_index:
                    w = interaction.weight
                    profile_vector += (
                        w * features_matrix[id_to_index[interaction.listing_id]]
                    )
                    total_weight += w
        if total_weight > 0:
            profile_vector /= total_weight
        return profile_vector, total_weight

    def _detect_preferred_location(self, interaction_features):
        if not interaction_features:
            return None, None, None, None
        provinces = [f.province_code for f in interaction_features if f.province_code]
        districts = [f.district_id for f in interaction_features if f.district_id]
        wards_id = [f.ward_id for f in interaction_features if f.ward_id]
        wards_code = [f.ward_code for f in interaction_features if f.ward_code]
        from collections import Counter

        pref_prov = Counter(provinces).most_common(1)[0][0] if provinces else None
        pref_dist = Counter(districts).most_common(1)[0][0] if districts else None
        pref_ward_id = Counter(wards_id).most_common(1)[0][0] if wards_id else None
        pref_ward_code = (
            Counter(wards_code).most_common(1)[0][0] if wards_code else None
        )
        return pref_prov, pref_dist, pref_ward_id, pref_ward_code

    def _build_feature_matrix(
        self, listings: List[ListingFeature]
    ) -> Tuple[np.ndarray, Dict[int, int]]:
        if not listings:
            return np.array([]), {}

        prices = np.array([listing.price for listing in listings]).reshape(-1, 1)
        areas = np.array([listing.area or 0.0 for listing in listings]).reshape(-1, 1)
        bedrooms = np.array([listing.bedrooms or 0 for listing in listings]).reshape(
            -1, 1
        )

        scaler = MinMaxScaler()
        prices_norm = scaler.fit_transform(prices)
        areas_norm = scaler.fit_transform(areas)
        bedrooms_norm = scaler.fit_transform(bedrooms)

        # One-hot encoding simple emulation
        product_types = ["ROOM", "APARTMENT", "HOUSE", "STUDIO", "OFFICE"]
        listing_types = ["RENT", "SALE", "SHARE"]

        # Location features
        province_codes = [listing.province_code or "UNKNOWN" for listing in listings]

        # Unique province codes for one-hot
        unique_provinces = list(set(province_codes))

        matrix = []
        id_to_index = {}
        for idx, listing in enumerate(listings):
            id_to_index[listing.listing_id] = idx

            # Base features: price, area, bedrooms (normalized)
            # Apply weights: Price 1.2x, Area 1.0x, Bedrooms 1.0x
            row = [
                prices_norm[idx][0] * 1.2,
                areas_norm[idx][0] * 1.0,
                bedrooms_norm[idx][0] * 1.0,
            ]

            # Add one-hot product type (Weight: 1.5x)
            row.extend(
                [1.5 if listing.product_type == pt else 0.0 for pt in product_types]
            )
            # Add one-hot listing type (Weight 1.0)
            row.extend(
                [1.0 if listing.listing_type == lt else 0.0 for lt in listing_types]
            )

            # Location features (one-hot or direct match)
            # We use a simple approach: if we have 100+ provinces, one-hot might be too wide.
            # But here we focus on the target vs candidates.
            # Let's add province match as a high-weight feature implicitly by the matrix
            # or we can handle it in the distance calculation.
            # For now, let's add them to the matrix to allow cosine similarity to see them.
            row.extend(
                [1.0 if province_codes[idx] == p else 0.0 for p in unique_provinces]
            )

            matrix.append(row)

        return np.array(matrix), id_to_index

    def _compute_cf_scores(
        self,
        user_interactions: List[InteractionEntry],
        all_interactions: List[InteractionEntry],
        candidate_ids: List[int],
    ) -> Dict[int, float]:
        # Build item-to-users index: { listing_id: { user_id: weight } }
        item_to_users: Dict[int, Dict[str, float]] = {}
        for interaction in all_interactions:
            u_id = interaction.user_id
            l_id = interaction.listing_id
            w = interaction.weight

            if l_id not in item_to_users:
                item_to_users[l_id] = {}
            item_to_users[l_id][u_id] = max(item_to_users[l_id].get(u_id, 0.0), w)

        # Target user items
        target_user_items = {i.listing_id: i.weight for i in user_interactions}
        if not target_user_items or not item_to_users:
            return {c: 0.0 for c in candidate_ids}

        # Item-Item co-occurrence
        cf_scores = {}
        for candidate_id in candidate_ids:
            if candidate_id in target_user_items:
                continue

            score = 0.0
            candidate_users = item_to_users.get(candidate_id, {})
            if not candidate_users:
                cf_scores[candidate_id] = 0.0
                continue

            for target_id, t_weight in target_user_items.items():
                target_users = item_to_users.get(target_id, {})
                if not target_users:
                    continue

                # Intersection of users who interacted with both candidate and target
                # dict.keys() in Python 3 supports set-like operations (&) implemented in C
                common_users = target_users.keys() & candidate_users.keys()
                if not common_users:
                    continue

                co_occur_users = sum(
                    min(target_users[u], candidate_users[u]) for u in common_users
                )
                score += co_occur_users * t_weight

            # Normalize score
            cf_scores[candidate_id] = min(score / max(1, len(target_user_items)), 1.0)

        return cf_scores
