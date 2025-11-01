import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


class TwoStageUncertaintyModel:
    """Two-stage model for price prediction with confidence intervals."""

    def __init__(
        self,
        model0,
        model1,
        n_splits=5,
        method="squared_error",
        seed=None,
        lower_bound=1e-6,
        alpha=0.1,
        gamma0=1.65,
        gamma1=1.75,
        features1=None,
    ):
        self.model0, self.model1 = model0, model1
        self.n_splits, self.method, self.seed = n_splits, method, seed
        self.gamma0, self.gamma1 = gamma0, gamma1
        self.lower_bound, self.alpha, self.features1 = lower_bound, alpha, features1
        self.fitted_ = False

    def _prepare_features_for_model1(self, X, y_pred):
        X_tmp = X[self.features1].copy() if self.features1 != "same" else X.copy()
        X_tmp["y_pred"] = y_pred
        return X_tmp

    def _get_target(self, y, oof_preds):
        return (
            (y - oof_preds) ** 2 + 1e-6
            if self.method == "squared_error"
            else np.abs(y - oof_preds)
        )

    def fit(self, X, y):
        y = np.asarray(y)
        oof_preds = np.zeros_like(y, dtype=float)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val, y_tr = X.iloc[train_idx], X.iloc[val_idx], y[train_idx]
            self.model0.fit(X_tr, y_tr)
            oof_preds[val_idx] = self.model0.predict(X_val)

        target = self._get_target(y, oof_preds)
        X_resid_feat = (
            self._prepare_features_for_model1(X, oof_preds)
            if self.features1
            else oof_preds.reshape(-1, 1)
        )
        self.model1.fit(X_resid_feat, target)
        self.model0.fit(X, y)
        self.fitted_ = True
        return self

    def predict_components(self, X):
        if not self.fitted_:
            raise ValueError("Call fit() before predict()")
        y_hat = self.model0.predict(X)
        X_resid_feat = (
            self._prepare_features_for_model1(X, y_hat)
            if self.features1
            else y_hat.reshape(-1, 1)
        )
        err_hat = self.model1.predict(X_resid_feat)
        err_hat = np.maximum(err_hat, self.lower_bound)
        return y_hat, err_hat

    def build_interval(self, y_hat, err_hat):
        err_hat_sqrt = np.sqrt(err_hat) if self.method == "squared_error" else err_hat
        lower = y_hat - self.gamma0 * err_hat_sqrt
        upper = y_hat + self.gamma1 * err_hat_sqrt
        return lower, upper

    def predict(self, X):
        y_hat, err_hat = self.predict_components(X)
        lower, upper = self.build_interval(y_hat, err_hat)
        return y_hat, lower, upper


class TwoStageDiverseEnsemble:
    """Ensemble of multiple TwoStageUncertaintyModel with different hyperparameters."""

    def __init__(self, model_configs, seed=None):
        """
        Args:
            model_configs: list of tuples (model0_instance, model1_instance)
        """
        self.model_configs = model_configs
        self.models = []
        self.seed = seed

    def fit(self, X, y):
        self.models = []
        for i, (m0, m1) in enumerate(self.model_configs):
            model = TwoStageUncertaintyModel(
                model0=m0,
                model1=m1,
                seed=(self.seed + i if self.seed is not None else None),
                features1="same",
            )
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X):
        y_hats, lowers, uppers = [], [], []
        for model in self.models:
            y_hat, lower, upper = model.predict(X)
            y_hats.append(y_hat)
            lowers.append(lower)
            uppers.append(upper)

        y_hat_ens = np.mean(y_hats, axis=0)
        lower_ens = np.mean(lowers, axis=0)
        upper_ens = np.mean(uppers, axis=0)
        return y_hat_ens, lower_ens, upper_ens


class RealEstatePricePredictorModel:
    """Main model for Vietnamese real estate price prediction."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.knn_scaler = StandardScaler()
        from sklearn.neighbors import NearestNeighbors

        self.knn_model = NearestNeighbors(n_neighbors=10)
        self.training_data = (
            None  # Store training data for KNN features during prediction
        )
        self.location_encoders = {}  # Store encoders for location columns
        self.is_fitted = False
        # Model hyperparameters
        self.SEED = 42
        self.xgb_params = {
            "n_estimators": 1000,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "random_state": self.SEED,
            "tree_method": "hist",
            "device": "cpu",
        }
        self.xgb_params1 = {
            "objective": "reg:gamma",
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "reg_alpha": 2.0,
            "reg_lambda": 5.0,
            "random_state": self.SEED,
            "tree_method": "hist",
            "device": "cpu",
        }

    def _preprocess_vietnamese_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Vietnamese real estate data from BatDongSan.com format
        """
        df = data.copy()

        # Handle date columns
        if "post_date" in df.columns:
            df["post_date"] = pd.to_datetime(df.post_date, errors="coerce")
            df["year"] = df["post_date"].dt.year
            df["month"] = df["post_date"].dt.month
            df.drop(["post_date"], axis=1, inplace=True)

        # Handle Vietnamese property types
        if "property_type" in df.columns:
            # Map Vietnamese property types to numeric
            property_type_mapping = {
                "Nhà trọ, phòng trọ": 1,
                "Chung cư": 2,
                "Cho thuê nhà trọ, phòng trọ": 3,
                "Văn phòng": 4,
                "Ký túc xá": 5,
            }
            df["property_type_encoded"] = (
                df["property_type"].map(property_type_mapping).fillna(0)
            )

        # Handle bedrooms - TẠMTHỜI BỎ QUA do data bị sai (bedrooms = price)
        # TODO: Cần data đúng với bedrooms thực tế
        # if 'bedrooms' in df.columns:
        #     df['bedroom'] = df['bedrooms'].fillna(0)

        # Handle area - TẠMTHỜI BỎ QUA do tất cả = 0
        # TODO: Cần data đúng với area thực tế
        # if 'area' in df.columns:
        #     df['area'] = df['area'].fillna(df['area'].median())

        # Handle location (Vietnamese addresses)
        location_cols = ["city", "district", "ward"]
        for col in location_cols:
            if col in df.columns:
                # Ép kiểu về string để tránh lỗi OrdinalEncoder
                df[col] = df[col].astype(str)
                # Sử dụng encoder riêng cho từng column
                if col not in self.location_encoders:
                    self.location_encoders[col] = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    df[f"{col}_encoded"] = (
                        self.location_encoders[col].fit_transform(df[[col]]).flatten()
                    )
                else:
                    # Khi dự đoán, chỉ transform
                    df[f"{col}_encoded"] = (
                        self.location_encoders[col].transform(df[[col]]).flatten()
                    )

        return df

    def _create_knn_features(
        self, df: pd.DataFrame, target_col: str = "price", is_training: bool = False
    ) -> pd.DataFrame:
        """
        Create KNN-based features for spatial-temporal patterns
        Sử dụng NearestNeighbors để tính khoảng cách và giá trung bình lân cận
        """
        from sklearn.neighbors import NearestNeighbors

        if not all(col in df.columns for col in ["latitude", "longitude", "year"]):
            return df

        knn_features = ["latitude", "longitude", "year"]

        if is_training:
            # Khi training: fit scaler và knn model với training data
            df_knn = df.dropna(subset=knn_features).copy()
            if len(df_knn) == 0:
                return df
            knn_data = self.knn_scaler.fit_transform(df_knn[knn_features])
            n_samples = knn_data.shape[0]
            if n_samples > 0:
                n_neighbors = min(10, n_samples)
            else:
                n_neighbors = 1
            self.knn_model = NearestNeighbors(n_neighbors=n_neighbors)
            self.knn_model.fit(knn_data)

            # Lưu lại training_knn_data để sử dụng khi dự đoán
            self.training_knn_data = df_knn.copy()

            distances, indices = self.knn_model.kneighbors(knn_data)
            if target_col in df_knn.columns:
                neighbor_prices = np.array(
                    [df_knn[target_col].values[inds] for inds in indices]
                )
                df_knn["price_knn"] = neighbor_prices.mean(axis=1)
            df_knn["k_dist"] = distances.mean(axis=1)
            # Gán lại các giá trị này về df gốc (theo index)
            for col in ["price_knn", "k_dist"]:
                if col in df_knn.columns:
                    df.loc[df_knn.index, col] = df_knn[col]
        else:
            # Khi dự đoán: sử dụng training data đã có để tìm hàng xóm cho new data
            if (
                self.training_data is None
                or self.knn_model is None
                or not hasattr(self, "training_knn_data")
            ):
                # Nếu chưa có training data, trả về df với NaN
                df["price_knn"] = np.nan
                df["k_dist"] = np.nan
                return df

            # Transform new data bằng scaler đã fit
            new_data = df[knn_features].copy()
            if new_data.isna().any().any():
                df["price_knn"] = np.nan
                df["k_dist"] = np.nan
                return df

            new_data_scaled = self.knn_scaler.transform(new_data)

            # Tìm hàng xóm gần nhất trong training data
            n_samples_fit = getattr(self.knn_model, "n_samples_fit_", 1)
            n_neighbors = min(10, n_samples_fit)
            distances, indices = self.knn_model.kneighbors(
                new_data_scaled, n_neighbors=n_neighbors
            )

            # Tính price_knn từ training data đã được lưu từ lúc fit
            if (
                len(self.training_knn_data) > 0
                and target_col in self.training_knn_data.columns
            ):
                neighbor_prices = np.array(
                    [
                        self.training_knn_data[target_col].values[inds]
                        for inds in indices
                    ]
                )
                df["price_knn"] = neighbor_prices.mean(axis=1)
            else:
                df["price_knn"] = np.nan

            df["k_dist"] = distances.mean(axis=1)

        return df

    def fit(self, train_data: pd.DataFrame, target_column: str = "price"):
        """
        Train the model with Vietnamese real estate data
        """
        # Preprocess data
        processed_data = self._preprocess_vietnamese_data(train_data)
        processed_data = self._create_knn_features(
            processed_data, target_column, is_training=True
        )

        # Store processed training data for KNN during prediction
        self.training_data = processed_data.copy()

        # Prepare features - using available and validated features
        feature_columns = [
            # Geographic coordinates
            "latitude",
            "longitude",
            # Temporal features
            "year",
            "month",
            # Property type
            "property_type_encoded",
            # Location details (crucial for Vietnamese real estate pricing)
            "city_encoded",
            "district_encoded",
            "ward_encoded",
            # Physical attributes - temporarily excluded due to data quality issues
            # 'area', 'bedroom',  # TODO: Need accurate data
            # KNN spatial features
            "price_knn",
            "k_dist",
        ]
        available_features = [
            col for col in feature_columns if col in processed_data.columns
        ]

        X = processed_data[available_features]
        y = processed_data[target_column]

        # Remove rows with NaN or invalid target values
        valid_idx = (~y.isna()) & (~X.isna().any(axis=1)) & (~pd.isnull(y))
        X = X[valid_idx]
        y = y[valid_idx]

        # Additionally remove infinite values
        import numpy as np

        valid_idx2 = (~np.isinf(y)) & (~np.isinf(X).any(axis=1))
        X = X[valid_idx2]
        y = y[valid_idx2]

        # Scale features
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Create ensemble model
        model_configs = [
            (XGBRegressor(**self.xgb_params), XGBRegressor(**self.xgb_params1)),
            (
                XGBRegressor(**{**self.xgb_params, "max_depth": 6}),
                XGBRegressor(**{**self.xgb_params1, "n_estimators": 800}),
            ),
            (
                XGBRegressor(**{**self.xgb_params, "learning_rate": 0.05}),
                XGBRegressor(**{**self.xgb_params1, "learning_rate": 0.07}),
            ),
        ]

        self.model = TwoStageDiverseEnsemble(model_configs, seed=self.SEED)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def load_and_train_from_sql(self, sql_file_path: str = None):
        """
        Load training data from SQL file and train the model

        Args:
            sql_file_path: Path to SQL file. If None, uses default path.
        """
        if sql_file_path is None:
            import os

            current_dir = os.path.dirname(os.path.abspath(__file__))
            sql_file_path = os.path.join(
                current_dir, "scraped_properties_202509012121.sql"
            )

        # Load data from SQL file
        df = load_data_from_sql_insert(sql_file_path)

        # Train model
        self.fit(df, target_column="price")

        return self

    def predict_price_range(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict price range for a single property

        Args:
            property_data: Dictionary containing property information

        Returns:
            Dictionary with predicted price range and confidence metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Convert to DataFrame
        df = pd.DataFrame([property_data])

        # Preprocess
        processed_data = self._preprocess_vietnamese_data(df)
        processed_data = self._create_knn_features(processed_data, is_training=False)

        # Prepare features - SỬ DỤNG CÙNG FEATURE LIST NHU KHI TRAIN
        feature_columns = [
            # Vị trí địa lý
            "latitude",
            "longitude",
            # Thời gian
            "year",
            "month",
            # Loại BĐS
            "property_type_encoded",
            # Địa chỉ chi tiết (rất quan trọng cho giá BĐS Việt Nam)
            "city_encoded",
            "district_encoded",
            "ward_encoded",
            # Thông số vật lý - TẠMTHỜI BỎ QUA do data sai
            # 'area', 'bedroom',  # TODO: Cần data đúng
            # KNN features
            "price_knn",
            "k_dist",
        ]
        available_features = [
            col for col in feature_columns if col in processed_data.columns
        ]
        X = processed_data[available_features]

        # Scale and predict
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        predicted_price, lower_bound, upper_bound = self.model.predict(X_scaled)

        return {
            "predicted_price": float(predicted_price[0]),
            "price_range": {"min": float(lower_bound[0]), "max": float(upper_bound[0])},
            "confidence_interval": 0.9,  # 90% confidence based on alpha=0.1
            "currency": "VND_millions",
        }

    def evaluate_price_vs_market(
        self, property_data: Dict[str, Any], asking_price: float
    ) -> Dict[str, Any]:
        """
        Đánh giá giá nhà so với mặt bằng thị trường (theo kiểu BatDongSan.com)

        Args:
            property_data: Dictionary containing property information
            asking_price: Giá user nhập vào (triệu VND)

        Returns:
            Dictionary với đánh giá chi tiết
        """
        # Dự đoán giá thị trường cho property này
        prediction = self.predict_price_range(property_data)
        predicted_price = prediction["predicted_price"]

        # Tính % chênh lệch
        price_diff_percent = ((asking_price - predicted_price) / predicted_price) * 100

        # Ngưỡng đánh giá dựa trên phân tích thị trường Việt Nam
        price_thresholds = {
            "very_low": (-float("inf"), -25),  # < -25%
            "low": (-25, -10),  # -25% đến -10%
            "reasonable": (-10, 15),  # -10% đến +15%
            "high": (15, 30),  # +15% đến +30%
            "very_high": (30, float("inf")),  # > +30%
        }

        # Xác định category
        category = "reasonable"
        for cat, (min_thresh, max_thresh) in price_thresholds.items():
            if min_thresh < price_diff_percent <= max_thresh:
                category = cat
                break

        # Messages by category
        messages = {
            "very_low": {
                "message": f"Price is very low compared to market. {abs(price_diff_percent):.1f}% below predicted price.",
                "recommendation": "Potential good investment opportunity, but verify property condition and legal status.",
                "category": "very_low",
            },
            "low": {
                "message": f"Price is below market rate. {abs(price_diff_percent):.1f}% below predicted price.",
                "recommendation": "Attractive price for the area. Consider quick action.",
                "category": "low",
            },
            "reasonable": {
                "message": f"Price is reasonable for the market. {price_diff_percent:+.1f}% difference from predicted price.",
                "recommendation": "Price is appropriate for current market conditions.",
                "category": "reasonable",
            },
            "high": {
                "message": f"Price is above market rate. {price_diff_percent:.1f}% above predicted price.",
                "recommendation": "Consider careful evaluation or price negotiation.",
                "category": "high",
            },
            "very_high": {
                "message": f"Price is very high compared to market. {price_diff_percent:.1f}% above predicted price.",
                "recommendation": "Price is too high. Investigate reasons or consider other options.",
                "category": "very_high",
            },
        }

        result = {
            "asking_price": asking_price,
            "predicted_price": predicted_price,
            "price_difference_percentage": price_diff_percent,
            "market_evaluation": {
                "category": category,
                "message": messages[category]["message"],
                "recommendation": messages[category]["recommendation"],
            },
            "price_range": prediction["price_range"],
            "confidence_interval": prediction["confidence_interval"],
            "currency": prediction["currency"],
        }

        return result

    def validate_price(
        self, property_data: Dict[str, Any], asking_price: float
    ) -> Dict[str, Any]:
        """
        Validate if asking price is within reasonable range (legacy method)

        Args:
            property_data: Property information
            asking_price: Asking price to validate

        Returns:
            Validation result with recommendation
        """
        prediction_result = self.predict_price_range(property_data)
        price_range = prediction_result["price_range"]

        is_valid = price_range["min"] <= asking_price <= price_range["max"]

        # Calculate price deviation
        predicted_price = prediction_result["predicted_price"]
        deviation_percent = abs(asking_price - predicted_price) / predicted_price * 100

        # Determine recommendation
        if is_valid:
            recommendation = "Price is within reasonable range"
        elif asking_price < price_range["min"]:
            recommendation = f"Price seems low. Consider increasing by {price_range['min'] - asking_price:.1f} million VND"
        else:
            recommendation = f"Price seems high. Consider reducing by {asking_price - price_range['max']:.1f} million VND"

        return {
            "is_valid": is_valid,
            "asking_price": asking_price,
            "predicted_price": predicted_price,
            "price_range": price_range,
            "deviation_percent": round(deviation_percent, 2),
            "recommendation": recommendation,
            "confidence": prediction_result["confidence_interval"],
        }

    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "encoder": self.encoder,
            "knn_scaler": self.knn_scaler,
            "knn_model": self.knn_model,
            "is_fitted": self.is_fitted,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.encoder = model_data["encoder"]
        self.knn_scaler = model_data["knn_scaler"]
        self.knn_model = model_data["knn_model"]
        self.is_fitted = model_data["is_fitted"]


def winkler_score(y_true, lower, upper, alpha=0.1, return_coverage=False):
    """
    Compute the Winkler Interval Score for prediction intervals.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower
    penalty_lower = 2 / alpha * (lower - y_true)
    penalty_upper = 2 / alpha * (y_true - upper)

    score = width.copy()
    score += np.where(y_true < lower, penalty_lower, 0)
    score += np.where(y_true > upper, penalty_upper, 0)

    if return_coverage:
        inside = (y_true >= lower) & (y_true <= upper)
        coverage = np.mean(inside)
        return np.mean(score), coverage

    return np.mean(score)


import ast
import csv

# --- Thêm hàm load_data_from_sql_insert để đọc dữ liệu từ file SQL dạng INSERT ---
import re


def load_data_from_sql_insert(filepath: str) -> pd.DataFrame:
    """
    Đọc file SQL dạng INSERT INTO ... VALUES (...),...; và trả về DataFrame.
    Sử dụng csv.reader để tách giá trị an toàn.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        sql = f.read()

    # Lấy tên cột
    col_match = re.search(r"\((.*?)\)\s*VALUES", sql, re.DOTALL)
    if not col_match:
        raise ValueError("Không tìm thấy tên cột trong file SQL")
    columns = [c.strip() for c in col_match.group(1).split(",")]

    # Tìm phần VALUES (...), (...), ...;
    match = re.search(r"VALUES\s*(.*);", sql, re.DOTALL)
    if not match:
        raise ValueError("Không tìm thấy VALUES trong file SQL")
    values_str = match.group(1)

    # Tách từng record (...), (...)
    records = re.findall(r"\((.*?)\)(?:,|$)", values_str, re.DOTALL)

    data = []
    for rec in records:
        # Dùng csv.reader để tách giá trị an toàn với dấu phẩy trong chuỗi
        reader = csv.reader([rec], delimiter=",", quotechar="'", skipinitialspace=True)
        vals = next(reader)
        row = []
        for v in vals:
            v = v.strip()
            if v.upper() == "NULL":
                row.append(None)
            else:
                try:
                    # Nếu là số
                    row.append(ast.literal_eval(v))
                except Exception:
                    row.append(v)
        data.append(row)

    # Lọc các record có đúng số cột
    filtered_data = [row for row in data if len(row) == len(columns)]
    if not filtered_data:
        raise ValueError(f"Không có record nào khớp số cột ({len(columns)})")

    df = pd.DataFrame(filtered_data, columns=columns)
    return df
