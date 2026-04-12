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

        # Feature engineering
        features_matrix, id_to_index = self._build_feature_matrix(
            [req.target] + req.candidates
        )
        target_vec = features_matrix[id_to_index[req.target.listing_id]]

        results = []
        for candidate in req.candidates:
            candidate_vec = features_matrix[id_to_index[candidate.listing_id]]
            # Cosine similarity (1D vectors -> reshape to 2D)
            cbf_score = cosine_similarity(
                target_vec.reshape(1, -1), candidate_vec.reshape(1, -1)
            )[0][0]

            # Boost based on VIP type and freshness
            vip_boost = self.VIP_WEIGHTS.get(candidate.vip_type.upper(), 1.0)
            freshness_boost = max(
                0, 1 - (candidate.post_date_days_ago * 0.01)
            )  # Decays over 100 days

            hybrid_score = cbf_score * vip_boost * (1 + 0.1 * freshness_boost)

            results.append(
                RecommendationItem(
                    listing_id=candidate.listing_id,
                    score=round(hybrid_score, 4),
                    cf_score=0.0,
                    cbf_score=round(cbf_score, 4),
                )
            )

        # Sort by VIP Status descending (Hard priority), then by hybrid score descending
        vip_rank_dict = {
            c.listing_id: self.VIP_RANKS.get(c.vip_type.upper(), 1)
            for c in req.candidates
        }
        results.sort(
            key=lambda x: (vip_rank_dict.get(x.listing_id, 1), x.score), reverse=True
        )
        return results[: req.top_n]

    async def get_personalized_feed(
        self, req: PersonalizedFeedRequest
    ) -> List[RecommendationItem]:
        if not req.candidates:
            return []

        # Feature matrix for candidates
        features_matrix, id_to_index = self._build_feature_matrix(req.candidates)

        profile_vector = np.zeros(features_matrix.shape[1])
        total_weight = 0.0

        # We don't have the full features of the user's past interacted listings here,
        # but the backend candidate list might contain some of them.
        for interaction in req.user_interactions:
            if interaction.listing_id in id_to_index:
                weight = interaction.weight
                idx = id_to_index[interaction.listing_id]
                profile_vector += weight * features_matrix[idx]
                total_weight += weight

        if total_weight > 0:
            profile_vector /= total_weight

        # CF Computation setup
        cf_scores = self._compute_cf_scores(
            req.user_interactions,
            req.all_interactions,
            [c.listing_id for c in req.candidates],
        )

        results = []
        for candidate in req.candidates:
            # CBF Score
            idx = id_to_index[candidate.listing_id]
            candidate_vec = features_matrix[idx]
            cbf_score = 0.0
            if total_weight > 0:
                cbf_score = cosine_similarity(
                    profile_vector.reshape(1, -1), candidate_vec.reshape(1, -1)
                )[0][0]

            # CF Score
            cf_score = cf_scores.get(candidate.listing_id, 0.0)

            # Combine
            vip_boost = self.VIP_WEIGHTS.get(candidate.vip_type.upper(), 1.0)
            freshness_boost = max(0, 1 - (candidate.post_date_days_ago * 0.01))

            # Hybrid
            alpha = (
                req.alpha if total_weight > 0 else 1.0
            )  # purely CF or default sort if no profile
            base_score = (alpha * cf_score) + ((1 - alpha) * cbf_score)
            final_score = base_score * vip_boost * (1 + 0.1 * freshness_boost)

            results.append(
                RecommendationItem(
                    listing_id=candidate.listing_id,
                    score=round(final_score, 4),
                    cf_score=round(cf_score, 4),
                    cbf_score=round(cbf_score, 4),
                )
            )

        # Sort by VIP Status descending (Hard priority), then by hybrid score descending
        vip_rank_dict = {
            c.listing_id: self.VIP_RANKS.get(c.vip_type.upper(), 1)
            for c in req.candidates
        }
        results.sort(
            key=lambda x: (vip_rank_dict.get(x.listing_id, 1), x.score), reverse=True
        )
        return results[: req.top_n]

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

        matrix = []
        id_to_index = {}
        for idx, listing in enumerate(listings):
            id_to_index[listing.listing_id] = idx

            row = [prices_norm[idx][0], areas_norm[idx][0], bedrooms_norm[idx][0]]

            # Add one-hot product type
            row.extend(
                [1.0 if listing.product_type == pt else 0.0 for pt in product_types]
            )
            # Add one-hot listing type
            row.extend(
                [1.0 if listing.listing_type == lt else 0.0 for lt in listing_types]
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
