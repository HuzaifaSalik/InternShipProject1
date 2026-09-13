"""Loading and cleaning.

Four defects are planted in this dataset and each is handled explicitly:

1. Negative weights (292 train / 145 validation) -- absolute values sit inside
   the normal 5,000-47,500 range, so this is sign corruption. Fixed with abs()
   and recorded in a flag.
2. Missing weight (300 / 165) -- imputed from the training median for that
   equipment type.
3. Missing market_index (374 / 249) -- imputed from the same calendar day.
   Benchmarked at 86% signal recovery against 0% for a global mean, because
   market_index is a property of the day rather than of the load.
4. Corrupted labels (~1.1% of training rows, inflated ~x3.9 or deflated ~x0.26)
   -- dropped from training only, never from the prediction sets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

RAW_NUMERIC = ["distance", "weight", "market_index", "quote_signal"]


def load_raw(path) -> pd.DataFrame:
    """Read one of the challenge CSVs with the date column parsed."""
    frame = pd.read_csv(path, parse_dates=["date"])
    for column in RAW_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _impute_market_index(frame: pd.DataFrame, global_fallback: float) -> pd.Series:
    """Fill market_index from the same date, then a widening window, then global.

    Uses only the market_index column of the frame it is given -- no labels are
    involved, so applying this to the validation file is not leakage.
    """
    values = frame["market_index"]
    daily = values.groupby(frame["date"]).transform("mean")
    filled = values.fillna(daily)

    if filled.isna().any():
        # Whole date missing: average the surrounding +/-N days.
        by_date = values.groupby(frame["date"]).mean().sort_index()
        window = config.MARKET_INDEX_WINDOW_DAYS * 2 + 1
        smoothed = by_date.rolling(window, center=True, min_periods=1).mean()
        filled = filled.fillna(frame["date"].map(smoothed))

    return filled.fillna(global_fallback)


class Cleaner:
    """Fits imputation statistics on training data, applies them anywhere.

    Only the weight medians are carried across files; market_index is filled
    per-frame from its own dates, which is both more accurate and free of any
    dependence on the training period.
    """

    def __init__(self) -> None:
        self.weight_median_by_equipment: dict[str, float] = {}
        self.weight_median_global: float = np.nan
        self.market_index_global: float = np.nan

    def fit(self, train: pd.DataFrame) -> "Cleaner":
        weight = train["weight"].abs()
        self.weight_median_by_equipment = (
            weight.groupby(train["equipment"]).median().to_dict()
        )
        self.weight_median_global = float(weight.median())
        self.market_index_global = float(train["market_index"].median())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()

        # 1 + 2: weight sign corruption, then missingness.
        out["weight_was_negative"] = (out["weight"] < 0).fillna(False).astype(int)
        out["weight_missing"] = out["weight"].isna().astype(int)
        weight = out["weight"].abs()
        fallback = out["equipment"].map(self.weight_median_by_equipment)
        out["weight"] = weight.fillna(fallback).fillna(self.weight_median_global)

        # 3: market_index missingness, filled from the same calendar day.
        out["market_index_missing"] = out["market_index"].isna().astype(int)
        out["market_index"] = _impute_market_index(out, self.market_index_global)

        return out


def flag_label_outliers(
    train: pd.DataFrame, sigma: float = config.OUTLIER_MAD_SIGMA
) -> pd.Series:
    """Boolean mask of training rows whose label looks corrupted.

    A global cut on rate-per-mile would be confounded by the distance effect
    (rate-per-mile falls with distance), so the cut is applied to residuals from
    a quick log-linear fit on log(distance) and equipment, scaled by the median
    absolute deviation.
    """
    log_rpm = np.log(train["posted_rate"] / train["distance"])
    design = [np.ones(len(train)), np.log(train["distance"].to_numpy(float))]
    for equipment in sorted(train["equipment"].unique())[1:]:
        design.append((train["equipment"] == equipment).to_numpy(float))
    matrix = np.column_stack(design)

    beta, *_ = np.linalg.lstsq(matrix, log_rpm.to_numpy(float), rcond=None)
    residual = log_rpm.to_numpy(float) - matrix @ beta

    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    scale = 1.4826 * mad
    return pd.Series(np.abs(residual - median) > sigma * scale, index=train.index)


def load_clean_train() -> tuple[pd.DataFrame, Cleaner, pd.Series]:
    """Training frame, the fitted cleaner, and the corrupted-label mask."""
    raw = load_raw(config.TRAIN_CSV)
    cleaner = Cleaner().fit(raw)
    train = cleaner.transform(raw)
    outliers = flag_label_outliers(train)
    return train, cleaner, outliers


def daily_market_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean market_index and quote_signal per calendar day.

    Used to give the December scenario rows the market conditions the chart
    inputs omit; validation.csv carries ~199 loads for every December day.
    """
    return (
        frame.groupby("date")[["market_index", "quote_signal"]]
        .mean()
        .rename(columns={"market_index": "market_index", "quote_signal": "quote_signal"})
    )
