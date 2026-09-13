"""Model families, all predicting log(rate per mile).

Rate per mile is the modelling target because distance alone explains ~91% of
raw-rate variance. Dividing it out leaves a homoscedastic target that matches
the multiplicative structure of the data, and exponentiating back guarantees the
strictly positive predictions score.py demands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config
from .features import LinearFeatures, TreeFeatures


def log_rate_per_mile(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame["posted_rate"].to_numpy(float) / frame["distance"].to_numpy(float))


class RateModel:
    """Common interface: fit on a cleaned frame, predict dollars."""

    def fit(self, frame: pd.DataFrame) -> "RateModel":  # pragma: no cover - interface
        raise NotImplementedError

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_rate(self, frame: pd.DataFrame) -> np.ndarray:
        rate = np.exp(self.predict_log_rpm(frame)) * frame["distance"].to_numpy(float)
        # score.py rejects non-positive predictions outright.
        return np.maximum(rate, 1.0)


class RidgeRateModel(RateModel):
    """Splines + geographic radial basis + a linear time trend.

    The primary candidate. It reaches 2.47% MAPE on the forward holdout without
    any trees, and it is the only family here that can extrapolate the upward
    drift past the end of the training period.
    """

    def __init__(self, alpha: float = 1.0, n_centers: int = config.GEO_CENTERS) -> None:
        self.alpha = alpha
        self.n_centers = n_centers
        self.features: LinearFeatures | None = None
        self.scaler = StandardScaler()
        self.ridge = Ridge(alpha=alpha, random_state=config.SEED)

    def fit(self, frame: pd.DataFrame) -> "RidgeRateModel":
        origin = frame["date"].min()
        self.features = LinearFeatures(origin, self.n_centers).fit(frame)
        matrix = self.scaler.fit_transform(self.features.transform(frame))
        self.ridge.fit(matrix, log_rate_per_mile(frame))
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.features is not None, "call fit first"
        return self.ridge.predict(self.scaler.transform(self.features.transform(frame)))


class TreeRateModel(RateModel):
    """Histogram gradient boosting on the raw drivers, with no date feature."""

    def __init__(self, **params) -> None:
        self.params = {
            "max_iter": 400,
            "learning_rate": 0.06,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": config.SEED,
            **params,
        }
        self.features: TreeFeatures | None = None
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, frame: pd.DataFrame) -> "TreeRateModel":
        self.features = TreeFeatures(frame["date"].min())
        self.model = HistGradientBoostingRegressor(
            categorical_features=self.features.categorical_mask, **self.params
        )
        self.model.fit(self.features.transform(frame), log_rate_per_mile(frame))
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.features is not None, "call fit first"
        return self.model.predict(self.features.transform(frame))


class HybridRateModel(RateModel):
    """Ridge carries level, trend and main effects; the GBM fits the residual.

    The structure of this dataset is close to additive in log space -- explicit
    interaction terms bought essentially nothing in testing -- so the residual
    stage is deliberately shrunk by `residual_weight`. If nothing is left for it
    to find, the weight can go to zero and no accuracy is lost.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        n_centers: int = config.GEO_CENTERS,
        residual_weight: float = 1.0,
        **tree_params,
    ) -> None:
        self.residual_weight = residual_weight
        self.linear = RidgeRateModel(alpha=alpha, n_centers=n_centers)
        self.tree = TreeRateModel(**tree_params)
        self._residual_model: HistGradientBoostingRegressor | None = None

    def fit(self, frame: pd.DataFrame) -> "HybridRateModel":
        self.linear.fit(frame)
        residual = log_rate_per_mile(frame) - self.linear.predict_log_rpm(frame)

        self.tree.features = TreeFeatures(frame["date"].min())
        self._residual_model = HistGradientBoostingRegressor(
            categorical_features=self.tree.features.categorical_mask, **self.tree.params
        )
        self._residual_model.fit(self.tree.features.transform(frame), residual)
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        assert self._residual_model is not None, "call fit first"
        assert self.tree.features is not None
        correction = self._residual_model.predict(self.tree.features.transform(frame))
        return self.linear.predict_log_rpm(frame) + self.residual_weight * correction


class MedianRateModel(RateModel):
    """Median rate-per-mile times distance. Reported as the floor to beat."""

    def __init__(self) -> None:
        self.value = 0.0

    def fit(self, frame: pd.DataFrame) -> "MedianRateModel":
        self.value = float(np.median(log_rate_per_mile(frame)))
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


class BlendRateModel(RateModel):
    """Equal-weight average of several fitted models, in log space.

    Averaging the top Optuna configurations rather than shipping the single
    winner is deliberate: the submission sits two months beyond the training
    period, and under that kind of shift the ranking of near-tied configurations
    is not stable enough to trust.
    """

    def __init__(self, specs: list[tuple[str, dict]]) -> None:
        self.specs = specs
        self.models: list[RateModel] = []

    def fit(self, frame: pd.DataFrame) -> "BlendRateModel":
        self.models = [build(name, **params).fit(frame) for name, params in self.specs]
        return self

    def predict_log_rpm(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.models, "call fit first"
        return np.mean([m.predict_log_rpm(frame) for m in self.models], axis=0)


def build(name: str, **params) -> RateModel:
    """Instantiate a model family by name (used by the Optuna study)."""
    builders = {
        "median": MedianRateModel,
        "ridge": RidgeRateModel,
        "tree": TreeRateModel,
        "hybrid": HybridRateModel,
        "blend": BlendRateModel,
    }
    if name not in builders:
        raise ValueError(f"unknown model family: {name}")
    return builders[name](**params)
