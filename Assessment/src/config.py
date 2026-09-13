"""Paths, constants and tunables shared across the pipeline."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
REPORT = ROOT / "report"

TRAIN_CSV = DATA / "train_test.csv"
VALID_CSV = DATA / "validation.csv"
TEMPLATE_CSV = DATA / "validation_predictions_template.csv"
DECEMBER_CSV = DATA / "december_chart_inputs.csv"

# score.py is invoked as `--predictions validation_predictions.csv` from the repo
# root, so the submission file lives there rather than under outputs/.
PREDICTIONS_CSV = ROOT / "validation_predictions.csv"
MODEL_PKL = OUTPUTS / "model.joblib"
METRICS_JSON = OUTPUTS / "cv_metrics.json"
STUDY_JSON = OUTPUTS / "optuna_study.json"

SEED = 42

# --- data cleaning -----------------------------------------------------------
# Labels are corrupted for ~1.1% of training rows (inflated ~x3.9 or deflated
# ~x0.26). Detected as residuals from a robust log-linear fit, not as raw
# rate-per-mile thresholds, so the cut adapts to the distance effect.
OUTLIER_MAD_SIGMA = 5.0

# market_index is a property of the *day*, not the load (within-date std 0.025
# against an across-date range of 0.685-1.468). Missing values are filled from
# the same date, then a widening window, then the global median.
MARKET_INDEX_WINDOW_DAYS = 3

# quote_signal is DELIBERATELY EXCLUDED from every feature set. It is not a
# market variable: it is the target itself, in one of three monthly regimes.
#   DIRECT   (Jan Feb Mar Jun Sep)  quote_signal == posted_rate/distance,
#                                   slope +1.000, R2 0.993
#   MIRRORED (Apr May Jul Oct)      quote_signal == 4.1487 - posted_rate/distance,
#                                   slope -1.000, R2 0.987
#   NOISE    (Aug, and Nov + Dec)   uninformative
# Pooling the months hides this: the direct and mirrored blocks cancel, leaving a
# pooled correlation of +0.05 that looks like a weak, harmless feature.
# The validation set (Nov-Dec) is entirely in the NOISE regime, detected without
# labels via corr(quote_signal, log distance): -0.80 DIRECT, +0.81 MIRRORED,
# ~0.0 NOISE; validation reads +0.009 (Nov) and +0.013 (Dec).
# Measured on the one labelled noise-regime month, training Jan-Jul -> test Aug:
#   keeping quote_signal   MAPE 2.359%
#   dropping quote_signal  MAPE 1.411%   <- 40% less error
# Any holdout inside the training range lands in a leaky month and will argue for
# keeping it. That argument does not transfer to the submission period.

# --- validation design -------------------------------------------------------
# Validation is 2025-11-01..2025-12-31, entirely after training ends 2025-10-31.
# The primary holdout reproduces that 2-month forward gap inside the training
# data: fit on Jan-Aug, score Sep-Oct.
HOLDOUT_START = "2025-09-01"

# Expanding-window folds: train on everything before `start`, score [start, end).
CV_FOLDS = [
    ("2025-06-01", "2025-07-01"),
    ("2025-07-01", "2025-08-01"),
    ("2025-08-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),  # the 2-month fold that mirrors the real task
]
# The final fold matches the submission horizon, so it carries extra weight in
# the Optuna objective.
CV_FOLD_WEIGHTS = [1.0, 1.0, 1.0, 2.0]

# --- December chart ----------------------------------------------------------
DEC_PICKUP = "Lexington"
DEC_DELIVERY = "Fort Wayne"

# --- feature engineering -----------------------------------------------------
SPLINE_KNOTS = {
    "log_distance": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],
    "market_index": [0.10, 0.25, 0.50, 0.75, 0.90],
    "log_weight": [0.20, 0.40, 0.60, 0.80],
}
GEO_CENTERS = 24
