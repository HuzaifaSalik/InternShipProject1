"""Feature engineering shared by training, cross-validation and prediction.

Two design decisions here are load-bearing and both were settled by measurement
on a forward-in-time holdout (fit Jan-Aug, score Sep-Oct):

Time. No date ordinal, month or day-of-year reaches the tree model. Every
validation row falls after the last training date, so a tree would place all
12,000 of them in the same terminal leaf and contribute a constant. Fourier
seasonal terms are worse than useless -- fitted on Jan-Aug they curl back down
and overshoot Sep-Oct by 4%. Only `days_since_start`, a single linear term in
the linear model, is allowed to carry the trend, because a slope extrapolates
and a split point cannot. Omitting time entirely costs 4.5% systematic bias.

Geography. Never city identity. 1,447 of the 12,000 validation rows involve a
city that never appears in training, which leaves city dummies and target
encodings undefined. A radial basis over coordinates scores the same (2.47% vs
2.45% MAPE) and degrades gracefully for unseen cities.

Leakage. quote_signal is excluded everywhere. It is the target in disguise and
its sign flips by month; the validation period carries no signal at all. The
full derivation and the measurements are in config.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from . import config

EARTH_RADIUS_MILES = 3958.7613

#: Columns handed to the tree model. Monotone transforms are redundant for
#: trees, so only the log forms are kept.
TREE_NUMERIC = [
    "log_distance",
    "market_index",
    "log_weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine",
    "circuity",
    "dx",
    "dy",
    "day_of_week",
    "is_weekend",
    "weight_missing",
    "market_index_missing",
    "weight_was_negative",
]

#: Linear terms entering the ridge model before spline and geographic bases.
LINEAR_NUMERIC = [
    "log_distance",
    "market_index",
    "log_weight",
    "circuity",
    "weight_missing",
    "market_index_missing",
    "weight_was_negative",
    "days_since_start",
]

EQUIPMENT_LEVELS = ["Dry Van", "Flatbed", "Reefer"]


def haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in miles.

    The supplied coordinates do not correspond to real US geography, but each
    city maps to exactly one consistent pair, so this remains a valid and
    useful metric in that coordinate space.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    inner = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def base_frame(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Model-ready columns derived from a cleaned raw frame."""
    out = pd.DataFrame(index=frame.index)

    out["log_distance"] = np.log(frame["distance"])
    out["log_weight"] = np.log(frame["weight"].clip(lower=1.0))
    # quote_signal is intentionally absent -- see the note in config.py. Not
    # reading it here also means december_chart_inputs.csv, which has no such
    # column, needs no special casing.
    out["market_index"] = frame["market_index"]

    out["pickup_lat"] = frame["pickup_lat"]
    out["pickup_lon"] = frame["pickup_lon"]
    out["delivery_lat"] = frame["delivery_lat"]
    out["delivery_lon"] = frame["delivery_lon"]

    straight_line = haversine_miles(
        frame["pickup_lat"], frame["pickup_lon"],
        frame["delivery_lat"], frame["delivery_lon"],
    )
    out["haversine"] = straight_line
    # Road miles per straight-line mile; a handful of rows are inconsistent and
    # the ratio makes that visible to the model rather than hiding it.
    out["circuity"] = frame["distance"] / np.maximum(straight_line, 1.0)
    # Direction of haul -- headhaul and backhaul price differently.
    out["dx"] = frame["delivery_lon"] - frame["pickup_lon"]
    out["dy"] = frame["delivery_lat"] - frame["pickup_lat"]

    out["day_of_week"] = frame["date"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["days_since_start"] = (frame["date"] - origin).dt.days / 365.0

    for column in ("weight_missing", "market_index_missing", "weight_was_negative"):
        out[column] = frame[column].astype(int) if column in frame else 0

    out["equipment"] = pd.Categorical(frame["equipment"], categories=EQUIPMENT_LEVELS)
    return out


class TreeFeatures:
    """Numeric matrix plus an ordinal equipment code for the GBM."""

    columns = TREE_NUMERIC + ["equipment_code"]
    categorical_mask = [False] * len(TREE_NUMERIC) + [True]

    def __init__(self, origin: pd.Timestamp) -> None:
        self.origin = origin

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        base = base_frame(frame, self.origin)
        matrix = base[TREE_NUMERIC].to_numpy(float)
        codes = base["equipment"].cat.codes.to_numpy(float).reshape(-1, 1)
        return np.hstack([matrix, codes])


class LinearFeatures:
    """Splines on the continuous drivers plus a radial basis over geography.

    Knots are placed at training quantiles and the geographic centres come from
    k-means over the training coordinates, so nothing here is hand-tuned to a
    particular split.
    """

    def __init__(self, origin: pd.Timestamp, n_centers: int = config.GEO_CENTERS) -> None:
        self.origin = origin
        self.n_centers = n_centers
        self.knots: dict[str, np.ndarray] = {}
        self.centers: np.ndarray | None = None
        self.geo_scale: float = 1.0
        self.names: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "LinearFeatures":
        base = base_frame(frame, self.origin)

        for column, quantiles in config.SPLINE_KNOTS.items():
            self.knots[column] = np.quantile(base[column].to_numpy(float), quantiles)

        coords = np.vstack([
            base[["pickup_lat", "pickup_lon"]].to_numpy(float),
            base[["delivery_lat", "delivery_lon"]].to_numpy(float),
        ])
        kmeans = KMeans(self.n_centers, n_init=10, random_state=config.SEED).fit(coords)
        self.centers = kmeans.cluster_centers_
        # Width set from the typical spacing between centres so the bases
        # overlap without washing out into a constant.
        spread = np.linalg.norm(coords - self.centers[kmeans.labels_], axis=1)
        self.geo_scale = float(max(np.mean(spread), 1e-3))
        return self

    def _radial(self, lat: np.ndarray, lon: np.ndarray) -> list[np.ndarray]:
        assert self.centers is not None
        points = np.column_stack([lat, lon])
        squared = ((points[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return list(np.exp(-squared / (2.0 * self.geo_scale**2)).T)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        base = base_frame(frame, self.origin)
        columns: list[np.ndarray] = []
        names: list[str] = []

        for column in LINEAR_NUMERIC:
            columns.append(base[column].to_numpy(float))
            names.append(column)

        for column, knots in self.knots.items():
            values = base[column].to_numpy(float)
            for knot in knots:
                columns.append(np.maximum(0.0, values - knot))
                names.append(f"{column}_hinge_{knot:.3f}")

        for level in EQUIPMENT_LEVELS[1:]:
            columns.append((base["equipment"] == level).to_numpy(float))
            names.append(f"equipment_{level}")

        for day in range(1, 7):
            columns.append((base["day_of_week"] == day).to_numpy(float))
            names.append(f"dow_{day}")

        for prefix, lat, lon in (
            ("pickup", base["pickup_lat"].to_numpy(float), base["pickup_lon"].to_numpy(float)),
            ("delivery", base["delivery_lat"].to_numpy(float), base["delivery_lon"].to_numpy(float)),
        ):
            for index, basis in enumerate(self._radial(lat, lon)):
                columns.append(basis)
                names.append(f"geo_{prefix}_{index}")

        self.names = names
        return np.column_stack(columns)
