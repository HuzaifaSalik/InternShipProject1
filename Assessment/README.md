# Freight Rate Prediction

Predicting posted freight rates for 12,000 loads covering 1 November – 31 December 2025,
trained on 48,000 labelled loads from 1 January – 31 October 2025.

The prediction window lies **entirely after** the training window, so this is a
forecasting problem rather than a random-split problem. Every design decision follows
from that.

## Results

| Metric | Value |
|---|---|
| MAE | **$51.12** |
| MAPE | **2.13%** |
| Bias | 0.982 |
| Median-rate baseline | $203.69 / 9.27% MAPE |
| Improvement over baseline | **4.4×** |

Measured on the September–October forward fold — the fold whose two-month horizon
matches the submission. Random K-fold on this dataset reports $37.16, understating
the true error by **27.3%**, and was never used for model selection.

## Key findings

1. **`quote_signal` leaks the target.** It is not a market variable. In five months it
   equals rate-per-mile exactly (slope +1.000, R² 0.993); in four months it is mirrored
   about a constant (`4.1487 − rate-per-mile`, slope −1.000, R² 0.987); in the remainder,
   including the **entire November–December prediction period**, it is noise. Pooled
   across months the two blocks cancel, hiding the effect behind a correlation of +0.05.
   Excluding the column cut error by roughly 40% on a like-for-like test. See the block
   comment in [`src/config.py`](src/config.py).

2. **Trees cannot extrapolate.** Every validation row falls after the last training date,
   so a tree places all 12,000 in one terminal leaf. Tree-only models underpredict by
   4–5%. A single linear term, `days_since_start`, carries the trend instead — removing it
   doubles the error.

3. **Four data defects were planted and handled explicitly:** negative weights (292 train
   / 145 validation), missing weights (300 / 165), missing `market_index` (374 / 249), and
   corrupted labels (677, 1.41% of training rows).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Requires Python 3.11+.

## Running the pipeline

```bash
# 1. Train: baselines, 60-trial Optuna search, selection, final fit
python -m src.train --trials 60 --blend-top 3

# 2. Generate both submission files
python -m src.predict

# 3. Validate output format and render the December chart
python score.py --predictions validation_predictions.csv \
                --december-predictions data/december_chart_inputs.csv

# 4. Rebuild the report from the latest metrics (optional)
python report/build_report.py
```

Step 1 takes several minutes (60 trials × 4 folds = 240 model fits). Use
`--trials 12` for a faster pass. All randomness is seeded (seed 42).

## Project structure

| Path | Purpose |
|---|---|
| `src/config.py` | Paths, seed, fold definitions, and the documented rationale for every constant |
| `src/data.py` | Loading, the four data-quality corrections, corrupted-label detection |
| `src/features.py` | Feature construction — two separate matrices for the linear and tree components |
| `src/model.py` | Model families behind one common interface |
| `src/evaluate.py` | Metrics, forward-in-time folds, and the random-split leakage contrast |
| `src/train.py` | Search, selection, final fit |
| `src/predict.py` | Generates both submission files |
| `score.py` | Provided by the assessor — validates format, renders the chart |
| `report/Freight_Rate_Report.docx` | Submission report |
| `report/MODELS.md` | Deep-dive on the three model families |
| `report/build_report.py` | Regenerates the report from `outputs/cv_metrics.json` |

## Approach

**Target.** `log(rate ÷ distance)`. Distance alone explains ~91% of raw-rate variance, so
dividing it out lets the model address the remaining variation; the logarithm matches the
multiplicative structure of freight pricing and guarantees strictly positive predictions
when exponentiated.

**Validation.** Chronological splits only. Four expanding-window folds, each training on
everything before a cut-off and scoring the window after it. The final fold spans two
months, matching the submission horizon, and carries double weight in the search
objective. Corrupted labels are removed from the training side of each split but retained
in the scored set, which is evaluated both ways.

**Model.** An equal-weight blend, in log space, of the three best hybrid configurations.
Each hybrid uses ridge regression — splines plus a radial basis over 40 geographic
centres plus a linear time trend — to carry the level and trend, with gradient-boosted
trees fitting the residual. The tree stage never sees a date feature, so it cannot break
the extrapolation.

**Geography** enters as coordinates, never city identity: 1,447 validation loads involve
a city absent from training, leaving dummies and target encodings undefined for those rows.

## Known limitations

- **Bias is not stable across folds** (0.983 to 1.033, changing sign). The trend term fits
  a straight line to a series that peaks in June, so a model trained through June
  overshoots July while one trained through August undershoots September–October. A fixed
  calibration factor cannot correct a sign-changing bias — tested, and it degraded results.
  Realistic expectation: ~2.1% MAPE with a level uncertainty of about ±2%.
- **No fully untouched holdout remains.** September–October serves both as the most heavily
  weighted fold in the search objective and as the set used to choose between the blend and
  the single configuration. The two differ by 0.5%, so the practical effect is negligible,
  but the headline figure is a tuned estimate rather than a clean out-of-sample one.

## Outputs

| File | Description |
|---|---|
| `validation_predictions.csv` | 12,000 rows of `load_id,predicted_rate` |
| `data/december_chart_inputs.csv` | 31 December scenario rows with `predicted_rate` filled |
| `scorer_results/candidate_december.png` | The fixed-input December chart |
| `outputs/cv_metrics.json` | Every metric from the training run |
| `outputs/model.joblib` | Fitted model and cleaner |
