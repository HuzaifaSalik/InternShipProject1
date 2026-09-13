"""Metrics and the forward-in-time validation harness.

Every validation row falls after the last training date, so the split has to be
chronological. Random K-fold is also computed, but only to demonstrate in the
report how optimistic it is on this dataset -- it lets a model learn from
November-adjacent patterns it would never have at submission time.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

from . import config


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Dollar-scale error. The graded metric is hidden, so report the family."""
    error = predicted - actual
    absolute_pct = np.abs(error) / actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(np.mean(absolute_pct) * 100),
        "median_ape": float(np.median(absolute_pct) * 100),
        "bias": float(np.mean(predicted) / np.mean(actual)),
        "n": int(len(actual)),
    }


def forward_folds(
    frame: pd.DataFrame, folds=config.CV_FOLDS
) -> Iterator[tuple[pd.Index, pd.Index, str]]:
    """Expanding window: train on everything before `start`, score [start, end)."""
    for start, end in folds:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        train_index = frame.index[frame["date"] < start_ts]
        test_index = frame.index[(frame["date"] >= start_ts) & (frame["date"] < end_ts)]
        if len(train_index) and len(test_index):
            yield train_index, test_index, f"{start_ts.date()}..{end_ts.date()}"


def primary_holdout(frame: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Fit Jan-Aug, score Sep-Oct -- the same 2-month gap as train to validation."""
    cutoff = pd.Timestamp(config.HOLDOUT_START)
    return frame.index[frame["date"] < cutoff], frame.index[frame["date"] >= cutoff]


def random_folds(
    frame: pd.DataFrame, n_splits: int = 4, seed: int = config.SEED
) -> Iterator[tuple[pd.Index, pd.Index, str]]:
    """Shuffled K-fold. Included for the leakage contrast, never for selection."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    for fold, chunk in enumerate(np.array_split(order, n_splits)):
        mask = np.zeros(len(frame), dtype=bool)
        mask[chunk] = True
        yield frame.index[~mask], frame.index[mask], f"random_fold_{fold + 1}"


def score_split(
    build_model: Callable[[], object],
    frame: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
    clean_mask: pd.Series | None = None,
) -> dict[str, float]:
    """Fit on clean training rows, score on the holdout.

    Corrupted labels are removed from the *training* side only. The holdout is
    scored twice: on clean rows, which measures the model, and on every row,
    which is what a hidden grader computing error against a corrupted answer key
    would actually see.
    """
    train = frame.loc[train_index]
    if clean_mask is not None:
        train = train.loc[clean_mask.loc[train_index]]

    model = build_model()
    model.fit(train)

    test = frame.loc[test_index]
    predicted = model.predict_rate(test)
    actual = test["posted_rate"].to_numpy(float)

    result = {f"all_{k}": v for k, v in metrics(actual, predicted).items()}
    if clean_mask is not None:
        keep = clean_mask.loc[test_index].to_numpy()
        result.update({f"clean_{k}": v for k, v in metrics(actual[keep], predicted[keep]).items()})
    return result


def cross_validate(
    build_model: Callable[[], object],
    frame: pd.DataFrame,
    clean_mask: pd.Series | None = None,
    folds=config.CV_FOLDS,
    weights=config.CV_FOLD_WEIGHTS,
) -> dict:
    """Expanding-window CV with the final, longest-horizon fold weighted up."""
    per_fold = []
    for (train_index, test_index, label), weight in zip(
        forward_folds(frame, folds), weights
    ):
        scores = score_split(build_model, frame, train_index, test_index, clean_mask)
        scores["fold"] = label
        scores["weight"] = weight
        scores["n_train"] = len(train_index)
        per_fold.append(scores)

    total = sum(f["weight"] for f in per_fold)
    key = "clean_mae" if "clean_mae" in per_fold[0] else "all_mae"
    objective = sum(f[key] * f["weight"] for f in per_fold) / total
    return {"folds": per_fold, "objective_mae": objective}
