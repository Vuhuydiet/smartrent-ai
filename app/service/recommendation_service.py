from typing import Dict, List, Tuple

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

    async def get_similar_listings(
        self, req: SimilarListingRequest
    ) -> List[RecommendationItem]:
        if not req.candidates:
            return []

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

        # --- Score each candidate ---
        results = []
        for candidate in req.candidates:
            candidate_vec = features_matrix[id_to_index[candidate.listing_id]]

            # 1. Similarity score
            similarity_score = self._compute_similarity_with_penalty(
                target_vec, candidate_vec, req.target, candidate
            )

            # 2. Personalization score
            personalization_score = 0.0
            if has_personalization:
                personalization_score = cosine_similarity(
                    profile_vector.reshape(1, -1), candidate_vec.reshape(1, -1)
                )[0][0]

            # 3. Blended & Boosted final score
            final_score = self._calculate_final_score(
                similarity_score, personalization_score, candidate, has_personalization
            )

            results.append(
                RecommendationItem(
                    listing_id=candidate.listing_id,
                    score=round(final_score, 4),
                    cf_score=round(personalization_score, 4),
                    cbf_score=round(similarity_score, 4),
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[: req.top_n]

    def _compute_similarity_with_penalty(
        self, target_vec, candidate_vec, target_feat, candidate_feat
    ) -> float:
        similarity_score = cosine_similarity(
            target_vec.reshape(1, -1), candidate_vec.reshape(1, -1)
        )[0][0]

        if candidate_feat.province_code != target_feat.province_code:
            similarity_score *= 0.1
        elif (
            target_feat.district_id
            and candidate_feat.district_id
            and candidate_feat.district_id != target_feat.district_id
        ):
            similarity_score *= 0.7
        elif (
            target_feat.ward_id
            and candidate_feat.ward_id
            and candidate_feat.ward_id != target_feat.ward_id
        ) or (
            target_feat.ward_code
            and candidate_feat.ward_code
            and candidate_feat.ward_code != target_feat.ward_code
        ):
            similarity_score *= 0.9
        return float(similarity_score)

    def _calculate_final_score(
        self, sim_score, pers_score, candidate, has_pers
    ) -> float:
        if has_pers:
            blended = (0.9 * sim_score) + (0.1 * pers_score)
        else:
            blended = sim_score

        vip_boost = self.VIP_WEIGHTS.get(candidate.vip_type.upper(), 1.0)
        freshness_boost = max(0, 1 - (candidate.post_date_days_ago * 0.01))
        return float(blended * vip_boost * (1 + 0.1 * freshness_boost))

    async def get_personalized_feed(
        self, req: PersonalizedFeedRequest
    ) -> List[RecommendationItem]:
        if not req.candidates:
            return []

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

        # Shift Detection
        (
            shift_prov,
            shift_dist,
            shift_ward_id,
            shift_ward_code,
        ) = self._detect_shift_location(
            req.interaction_features, pref_prov, pref_dist, pref_ward_id, pref_ward_code
        )

        # CF Computation
        cf_scores = self._compute_cf_scores(
            req.user_interactions or [],
            req.all_interactions or [],
            [c.listing_id for c in req.candidates],
        )

        results = []
        for candidate in req.candidates:
            score_data = self._score_personalized_candidate(
                candidate,
                features_matrix[id_to_index[candidate.listing_id]],
                profile_vec,
                total_w,
                cf_scores.get(candidate.listing_id, 0.0),
                pref_prov,
                pref_dist,
                pref_ward_id,
                pref_ward_code,
                shift_prov,
                shift_dist,
                shift_ward_id,
                shift_ward_code,
            )
            results.append(score_data)

        results.sort(key=lambda x: x.score, reverse=True)

        # Position Pinning
        # If there is a shift, find the top 3 candidates that match the shift location and pin them to pos 8,9,10
        if shift_ward_id or shift_ward_code or shift_dist or shift_prov:
            pinned: List[RecommendationItem] = []
            non_pinned: List[RecommendationItem] = []
            for item in results:
                c = next(
                    cand
                    for cand in req.candidates
                    if cand.listing_id == item.listing_id
                )
                match = False
                if shift_ward_id and c.ward_id == shift_ward_id:
                    match = True
                elif shift_ward_code and c.ward_code == shift_ward_code:
                    match = True
                elif (
                    shift_dist
                    and not shift_ward_id
                    and not shift_ward_code
                    and c.district_id == shift_dist
                ):
                    match = True
                elif (
                    shift_prov
                    and not shift_dist
                    and not shift_ward_id
                    and not shift_ward_code
                    and c.province_code == shift_prov
                ):
                    match = True

                if match and len(pinned) < 3:
                    pinned.append(item)
                else:
                    non_pinned.append(item)

            final_results = non_pinned[:7] + pinned + non_pinned[7:]
            return final_results[: req.top_n]

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

    def _score_personalized_candidate(
        self,
        candidate,
        candidate_vec,
        profile_vec,
        total_w,
        cf_score,
        pref_prov,
        pref_dist,
        pref_ward_id,
        pref_ward_code,
        shift_prov,
        shift_dist,
        shift_ward_id,
        shift_ward_code,
    ) -> RecommendationItem:
        # 1. CBF Score
        cbf_score = 0.0
        if total_w > 0:
            cbf_score = cosine_similarity(
                profile_vec.reshape(1, -1), candidate_vec.reshape(1, -1)
            )[0][0]

        # Geographic Penalty (uses preferred location unless there's a shift)
        target_prov = shift_prov or pref_prov
        target_dist = shift_dist or pref_dist
        target_ward_id = shift_ward_id or pref_ward_id
        target_ward_code = shift_ward_code or pref_ward_code

        if target_prov and candidate.province_code != target_prov:
            cbf_score *= 0.5
        elif target_dist and candidate.district_id != target_dist:
            cbf_score *= 0.7
        elif (
            target_ward_id and candidate.ward_id and candidate.ward_id != target_ward_id
        ) or (
            target_ward_code
            and candidate.ward_code
            and candidate.ward_code != target_ward_code
        ):
            cbf_score *= 0.9

        # Feature Weighting is applied via the _build_feature_matrix (weights were applied conceptually there,
        # or we can apply it on the base cbf score if product_type matches but matrix takes care of one-hot similarity)
        # To specifically boost product_type and price:
        # Product type one-hot is in the matrix, price is in the matrix.
        # We can add explicit boost if we know what product type is preferred.
        # The user profile vector naturally has higher values for the mode product type.

        # 2. Hybrid Base
        base_score = (
            (0.4 * cf_score) + (0.6 * cbf_score) if cf_score > 0 else cbf_score * 0.9
        )

        # 3. Boosts
        vip_boost = self.VIP_WEIGHTS.get(candidate.vip_type.upper(), 1.0)
        freshness_boost = max(0, 1 - (candidate.post_date_days_ago * 0.01))
        final_score = base_score * vip_boost * (1 + 0.1 * freshness_boost)

        return RecommendationItem(
            listing_id=candidate.listing_id,
            score=round(float(final_score), 4),
            cf_score=round(float(cf_score), 4),
            cbf_score=round(float(cbf_score), 4),
        )

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

    def _detect_shift_location(
        self, interaction_features, pref_prov, pref_dist, pref_ward_id, pref_ward_code
    ):
        if not interaction_features or len(interaction_features) == 0:
            return None, None, None, None

        weights = [0.5, 0.3, 0.2]
        recent_features = interaction_features[:3]

        prov_weights = {}
        dist_weights = {}
        ward_id_weights = {}
        ward_code_weights = {}

        for i, f in enumerate(recent_features):
            w = weights[i] if i < len(weights) else 0.0
            if f.province_code:
                prov_weights[f.province_code] = (
                    prov_weights.get(f.province_code, 0.0) + w
                )
            if f.district_id:
                dist_weights[f.district_id] = dist_weights.get(f.district_id, 0.0) + w
            if f.ward_id:
                ward_id_weights[f.ward_id] = ward_id_weights.get(f.ward_id, 0.0) + w
            if f.ward_code:
                ward_code_weights[f.ward_code] = (
                    ward_code_weights.get(f.ward_code, 0.0) + w
                )

        # Highest precision first: Ward > District > Province
        for ward_id, weight in ward_id_weights.items():
            if weight > 0.7 and ward_id != pref_ward_id:
                ward_code = next(
                    (f.ward_code for f in recent_features if f.ward_id == ward_id), None
                )
                return None, None, ward_id, ward_code

        for ward_code, weight in ward_code_weights.items():
            if weight > 0.7 and ward_code != pref_ward_code:
                ward_id = next(
                    (f.ward_id for f in recent_features if f.ward_code == ward_code),
                    None,
                )
                return None, None, ward_id, ward_code

        for district_id, weight in dist_weights.items():
            if weight > 0.7 and district_id != pref_dist:
                return None, district_id, None, None

        for province_code, weight in prov_weights.items():
            if weight > 0.7 and province_code != pref_prov:
                return province_code, None, None, None

        return None, None, None, None

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
        # Build user-item matrix from all_interactions
        user_item_matrix: Dict[
            str, Dict[int, float]
        ] = {}  # { user_id: { listing_id: weight } }
        for interaction in all_interactions:
            u_id = interaction.user_id
            l_id = interaction.listing_id
            w = interaction.weight

            if u_id not in user_item_matrix:
                user_item_matrix[u_id] = {}
            # Keep max weight if multiple interactions
            user_item_matrix[u_id][l_id] = max(user_item_matrix[u_id].get(l_id, 0.0), w)

        # Target user items
        target_user_items = {i.listing_id: i.weight for i in user_interactions}
        if not target_user_items or not user_item_matrix:
            return {c: 0.0 for c in candidate_ids}

        # Item-Item co-occurrence
        cf_scores = {}
        for candidate_id in candidate_ids:
            if candidate_id in target_user_items:
                continue  # User already interacted with this, backend should have filtered it, but just in case

            score = 0.0
            for target_id, t_weight in target_user_items.items():
                co_occur_users = 0.0
                for u_id, items in user_item_matrix.items():
                    if target_id in items and candidate_id in items:
                        co_occur_users += min(items[target_id], items[candidate_id])

                # Adding normalized co-occurrence score
                score += co_occur_users * t_weight

            # Normalize score
            cf_scores[candidate_id] = min(score / max(1, len(target_user_items)), 1.0)

        return cf_scores
