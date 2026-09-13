# The Three Models: Ridge, Tree and Hybrid

A walkthrough of [`src/model.py`](../src/model.py) and the feature code in
[`src/features.py`](../src/features.py) that feeds it.

All measurements in this document were produced by running the code, not copied
from its docstrings. Where a docstring disagrees with a measurement, the
measurement is quoted and the discrepancy noted.

---

## Table of contents

1. [The shared foundation](#1-the-shared-foundation)
2. [Two feature matrices, not one](#2-two-feature-matrices-not-one)
3. [RidgeRateModel](#3-ridgeratemodel)
4. [TreeRateModel](#4-treeratemodel)
5. [HybridRateModel](#5-hybridratemodel)
6. [BlendRateModel and MedianRateModel](#6-blendratemodel-and-medianratemodel)
7. [The `build()` factory](#7-the-build-factory)
8. [A worked example, end to end](#8-a-worked-example-end-to-end)
9. [Measured results](#9-measured-results)
10. [Summary](#10-summary)

---

## 1. The shared foundation

### 1.1 What all three models actually predict

None of the three models predicts a dollar price directly. They all predict:

```python
def log_rate_per_mile(frame):
    return np.log(frame["posted_rate"] / frame["distance"])
```

That is **two transformations stacked**, and each one earns its place.

**Step 1 — divide by distance.** Distance alone explains roughly 91% of the
variance in the raw rate. Asking a model to predict "$4,800" for a 2,400-mile
haul is asking it to learn multiplication, which is not the interesting part.
Dividing distance out leaves *rate per mile* — around $2.00–$2.50 — and the model
can spend all of its capacity on the genuinely hard 9%.

**Step 2 — take the logarithm.** Freight pricing is *multiplicative*. A reefer
costs about 20% more than a dry van; it does not cost $200 more regardless of
haul length. Logs turn multiplication into addition:

```
log(a × b) = log(a) + log(b)
```

which is exactly the shape a linear model can represent. It also stabilises the
variance — errors become roughly the same size everywhere on the scale
(*homoscedasticity*), which is what least-squares fitting assumes.

> **Jargon:** *homoscedastic* = the spread of errors is constant across the range
> of the data. The opposite (*heteroscedastic*) is when big values have big errors
> and small values have small errors — which is exactly what raw dollar rates do,
> and exactly what the log fixes.

### 1.2 Converting back to dollars

```python
class RateModel:
    def predict_rate(self, frame):
        rate = np.exp(self.predict_log_rpm(frame)) * frame["distance"]
        return np.maximum(rate, 1.0)
```

`np.exp` can never return a negative number, so **predictions are structurally
positive**. This matters: [`score.py`](../score.py) rejects the entire submission
if any predicted rate is `<= 0`. Rather than checking for that failure and
patching it up, the design makes it impossible. The `np.maximum(rate, 1.0)` is
belt-and-braces on top.

#### A subtlety worth knowing: retransformation bias

Taking `exp()` of an average-in-log-space gives you the **median**, not the
**mean**. This is *Jensen's inequality*, and in many log-target models it causes a
systematic underprediction that needs a *Duan smearing correction*.

Measured here: the log-residual variance is **0.00024** and the smearing factor is
**1.0001** — completely negligible. No correction is needed. It is mentioned only
because in most projects of this shape, it *is* a real problem.

### 1.3 The common interface

```python
class RateModel:
    def fit(self, frame) -> "RateModel":        raise NotImplementedError
    def predict_log_rpm(self, frame) -> array:  raise NotImplementedError
    def predict_rate(self, frame) -> array:     # concrete, shared by all
```

Every model takes a **cleaned pandas DataFrame** and returns a **NumPy array of
dollars**. Subclasses only implement `fit` and `predict_log_rpm`; the dollar
conversion is written once. This is why [`evaluate.py`](../src/evaluate.py) can
score any family with the same code path, and why
[`train.py`](../src/train.py) can search over the model family itself as if it
were just another hyperparameter.

`fit()` returns `self` so calls can be chained: `build("ridge").fit(train)`.

---

## 2. Two feature matrices, not one

This is the design decision most people miss on a first read. **The ridge model
and the tree model are fed genuinely different inputs.** Both start from the same
helper, `base_frame()`, then diverge.

### 2.1 `base_frame()` — the shared raw material

```python
def base_frame(frame, origin):
    out["log_distance"] = np.log(frame["distance"])
    out["log_weight"]   = np.log(frame["weight"].clip(lower=1.0))
    out["market_index"] = frame["market_index"]

    straight_line   = haversine_miles(pickup_lat, pickup_lon, delivery_lat, delivery_lon)
    out["haversine"] = straight_line
    out["circuity"]  = frame["distance"] / np.maximum(straight_line, 1.0)
    out["dx"] = delivery_lon - pickup_lon
    out["dy"] = delivery_lat - pickup_lat

    out["day_of_week"]      = frame["date"].dt.dayofweek
    out["is_weekend"]       = (out["day_of_week"] >= 5).astype(int)
    out["days_since_start"] = (frame["date"] - origin).dt.days / 365.0
    ...
```

| Feature | Meaning | Why it exists |
|---|---|---|
| `log_distance` | ln(miles) | Matches the multiplicative structure |
| `log_weight` | ln(pounds), floored at 1 | Same; `clip` guards `log(0)` |
| `haversine` | Great-circle distance | Straight-line separation of the two cities |
| `circuity` | `distance / haversine` | How indirect the route is. 1.0 = straight shot, 1.4 = lots of detour. Also exposes rows where the two disagree instead of hiding them |
| `dx`, `dy` | Longitude/latitude delta | **Direction** of haul — headhaul and backhaul price differently |
| `day_of_week` | 0=Mon … 6=Sun | Weekly rhythm in freight demand |
| `days_since_start` | Years since first training date | The time trend — see below |

> **Note:** `quote_signal` was removed from this function. It is not a market
> variable — it is the target in disguise, with a sign that flips by month, and it
> carries no signal at all in the November–December submission period. The full
> derivation is documented in [`config.py`](../src/config.py).

### 2.2 The critical asymmetry: time

```python
TREE_NUMERIC   = [..., "day_of_week", "is_weekend", ...]           # NO days_since_start
LINEAR_NUMERIC = [..., "days_since_start"]                          # YES
```

**`days_since_start` is given to the ridge model and deliberately withheld from
the tree.** The reason is fundamental, not a tuning preference.

A decision tree works by splitting on thresholds: *"is date < 2025-06-15?"*.
Every validation row falls **after** every training date. So all 12,000 validation
rows would land in the same rightmost leaf and receive an identical constant.
**Trees cannot extrapolate.** They can only interpolate between values they have
already seen.

A linear term can. `slope × days_since_start` keeps rising past the edge of the
training data. A slope extrapolates; a split point cannot.

Measured contribution (ridge, fit Jan–Aug, score Sep–Oct):

| | MAE | MAPE | bias |
|---|---|---|---|
| with `days_since_start` | $52.04 | 2.19% | 0.985 |
| without | $111.40 | 4.72% | 0.953 |

Removing it **doubles the error** and introduces a 4.7% systematic
underprediction. This single term is the most consequential modelling decision in
the project.

⚠️ **But it is also the most fragile.** See [§9.3](#93-the-stability-warning).

### 2.3 `TreeFeatures` — 17 columns

```python
class TreeFeatures:
    columns          = TREE_NUMERIC + ["equipment_code"]
    categorical_mask = [False] * len(TREE_NUMERIC) + [True]

    def transform(self, frame):
        base   = base_frame(frame, self.origin)
        matrix = base[TREE_NUMERIC].to_numpy(float)
        codes  = base["equipment"].cat.codes.to_numpy(float).reshape(-1, 1)
        return np.hstack([matrix, codes])
```

Output shape: **(n_rows, 17)** — 16 numeric columns plus one equipment code.

Equipment becomes a single integer (`Dry Van`→0, `Flatbed`→1, `Reefer`→2) rather
than three separate 0/1 columns. `categorical_mask` tells scikit-learn which
column to treat as categorical, so it splits on *set membership* (`equipment ∈
{Flatbed, Reefer}`) rather than on a meaningless numeric threshold
(`equipment_code < 1.5`).

No splines, no radial basis, no scaling. Trees are **invariant to any monotone
transform** — splitting on `x` at 100 is identical to splitting on `log(x)` at
`log(100)` — so all that machinery would be wasted work.

> ⚠️ **Gotcha:** `columns` and `categorical_mask` are **class attributes**,
> evaluated once at import time from `TREE_NUMERIC`. Mutating `TREE_NUMERIC` at
> runtime desynchronises the mask from the matrix and scikit-learn raises
> `ValueError: categorical_features set as a boolean mask must have shape
> (n_features,)`. Edit the module source, not the list object.

### 2.4 `LinearFeatures` — 112 columns

A linear model needs every non-linearity **built by hand**. The transform emits
four blocks:

```
  8 raw linear terms       LINEAR_NUMERIC
 16 spline hinges          log_distance(7) + market_index(5) + log_weight(4)
  2 equipment dummies      Flatbed, Reefer  (Dry Van is the baseline)
  6 day-of-week dummies    Tue..Sun         (Monday is the baseline)
 80 geographic bases       2 ends × 40 centres
───
112 columns total          (with n_centers=40, the value Optuna selected)
```

#### Splines: bending a straight line

```python
for knot in knots:
    columns.append(np.maximum(0.0, values - knot))
```

`max(0, x - knot)` is a **hinge** (the same function as a ReLU in neural
networks). Below the knot it is exactly 0 and contributes nothing; above it, it
grows linearly.

Small example. Suppose `log_distance` has a knot at 6.0, and the model has learnt
a base slope of `-0.30` plus a hinge coefficient of `+0.12`:

| `log_distance` | hinge value | total slope in effect |
|---|---|---|
| 5.5 | `max(0, -0.5)` = 0 | −0.30 |
| 6.0 | `max(0, 0.0)` = 0 | −0.30 |
| 6.5 | `max(0, 0.5)` = 0.5 | −0.30 + 0.12 = **−0.18** |

The line **bends** at the knot. Seven knots on `log_distance` give seven possible
bend points, enough to trace a smooth curve while remaining a linear model —
which means it still extrapolates, unlike a tree.

Knots are placed at **training quantiles** (`np.quantile`), not at fixed values,
so they automatically land where the data actually is:

```python
SPLINE_KNOTS = {
    "log_distance": [0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95],
    "market_index": [0.10, 0.25, 0.50, 0.75, 0.90],
    "log_weight":   [0.20, 0.40, 0.60, 0.80],
}
```

On the real data this produces knots at `log_distance` = 5.788, 6.293, 6.664,
6.942, … (i.e. roughly 327, 541, 784, 1035 miles).

#### The geographic radial basis: how a linear model learns a map

This is the cleverest part of the feature code.

```python
kmeans       = KMeans(self.n_centers, n_init=10, random_state=config.SEED).fit(coords)
self.centers = kmeans.cluster_centers_
spread       = np.linalg.norm(coords - self.centers[kmeans.labels_], axis=1)
self.geo_scale = float(max(np.mean(spread), 1e-3))

def _radial(self, lat, lon):
    points  = np.column_stack([lat, lon])
    squared = ((points[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
    return list(np.exp(-squared / (2.0 * self.geo_scale ** 2)).T)
```

**In plain English:** scatter 40 landmarks across the map (chosen by k-means, so
they land where the cities actually are). For each city, record how close it is to
each landmark as a number between 0 and 1 that fades smoothly with distance.

| City is… | Basis value |
|---|---|
| sitting on landmark 7 | ≈ 1.0 |
| moderately near landmark 7 | ≈ 0.6 |
| far from landmark 7 | ≈ 0.0 |

So instead of a city *name* the model receives 40 smooth numbers describing
*where* it is. The payoff:

- **1,447 of the 12,000 validation rows involve a city that never appears in
  training.** One-hot encoding or target encoding would leave those undefined.
- An unseen city still has coordinates, so it still gets sensible basis values. It
  **degrades gracefully** instead of breaking.
- Measured cost versus city identity: 2.47% vs 2.45% MAPE — statistically nothing.

`geo_scale` is derived from the *measured* average spread of points around their
assigned centre. Too narrow and every basis is 0; too wide and every basis is 1.
Deriving it from data avoids hand-tuning.

> **Note on `KMeans`:** it is fitted on the **stacked** pickup *and* delivery
> coordinates, so a single shared set of landmarks describes both ends. The two
> ends then each get their own 40 columns (`geo_pickup_*`, `geo_delivery_*`),
> letting the model price origin and destination separately.

---

## 3. RidgeRateModel

```python
class RidgeRateModel(RateModel):
    def __init__(self, alpha=1.0, n_centers=config.GEO_CENTERS):
        self.features = None
        self.scaler   = StandardScaler()
        self.ridge    = Ridge(alpha=alpha, random_state=config.SEED)

    def fit(self, frame):
        origin        = frame["date"].min()
        self.features = LinearFeatures(origin, self.n_centers).fit(frame)
        matrix        = self.scaler.fit_transform(self.features.transform(frame))
        self.ridge.fit(matrix, log_rate_per_mile(frame))
        return self

    def predict_log_rpm(self, frame):
        return self.ridge.predict(self.scaler.transform(self.features.transform(frame)))
```

### What ridge regression is

Ordinary least squares finds coefficients minimising squared error. **Ridge** adds
a penalty on the size of those coefficients:

```
minimise:   Σ(actual − predicted)²   +   alpha × Σ(coefficient²)
                  ↑ fit the data          ↑ keep coefficients small
```

`alpha` controls the trade-off. `alpha = 0` is plain OLS. Large `alpha` forces
coefficients toward zero, producing a smoother, more conservative model.

**Why it is needed here:** there are 112 columns, and many are highly correlated —
40 overlapping geographic bases, 7 hinges carved out of the same variable. With
correlated inputs, OLS can produce wild coefficients that cancel each other
(+5,000 on one basis, −4,998 on its neighbour). That fits the training data and
falls apart on anything new. Ridge prevents it.

Optuna searched `alpha` over `1e-2` to `1e3` (log scale) and selected **861.7** —
near the top of the range. **The data is asking for heavy regularisation.** That
is a meaningful signal: with the submission two months beyond the training
window, a smooth conservative fit generalises better than a sharp one.

### Why `StandardScaler` is mandatory here

```python
matrix = self.scaler.fit_transform(self.features.transform(frame))
```

The ridge penalty `Σ(coefficient²)` is **not scale-invariant**. Raw columns span
wildly different ranges — `log_distance` ≈ 6, `market_index` ≈ 1.0, geographic
bases ∈ [0,1], `days_since_start` ∈ [0,1]. A variable measured in large units
needs a small coefficient to have the same effect, so it would be penalised less.
Standardising (subtract mean, divide by standard deviation) puts every column on
equal footing so `alpha` means the same thing everywhere.

Note `fit_transform` during training versus `transform` at prediction time — the
means and standard deviations are learnt **once on training data** and merely
applied later. Using `fit_transform` at prediction time would be a data leak.

### Why ridge is the only family that can extrapolate

It is the only one that receives `days_since_start`, and it uses it as a **slope**.
When December arrives, `days_since_start` is simply a larger number and the
prediction continues along the fitted line. Neither the tree nor any split-based
method can do this.

### `random_state` on `Ridge`

`Ridge(alpha=alpha, random_state=config.SEED)` — this parameter only has an effect
when `solver='sag'` or `'saga'`. The default `solver='auto'` resolves to a
deterministic solver here, so the argument is inert. Verified: no warning is
raised on scikit-learn 1.9.0. Harmless, but it does not do anything.

---

## 4. TreeRateModel

```python
class TreeRateModel(RateModel):
    def __init__(self, **params):
        self.params = {
            "max_iter":          400,
            "learning_rate":     0.06,
            "max_leaf_nodes":    31,
            "min_samples_leaf":  40,
            "l2_regularization": 1.0,
            "early_stopping":    False,
            "random_state":      config.SEED,
            **params,
        }

    def fit(self, frame):
        self.features = TreeFeatures(frame["date"].min())
        self.model    = HistGradientBoostingRegressor(
            categorical_features=self.features.categorical_mask, **self.params
        )
        self.model.fit(self.features.transform(frame), log_rate_per_mile(frame))
        return self
```

### What gradient boosting is

Not one tree — **hundreds of small trees built in sequence**, each correcting the
previous ones' mistakes:

```
tree 1 makes a rough prediction
   → measure what is left over (the residual)
tree 2 is trained to predict that residual
   → measure what is STILL left over
tree 3 is trained to predict that
   ... 400 times
final prediction = tree1 + lr×tree2 + lr×tree3 + ...
```

**`HistGradientBoosting`** is the *histogram* variant: continuous features are
bucketed into 255 bins before searching for splits, so each split is chosen by
scanning 255 candidates instead of 48,000 values. This is what makes it fast —
it is scikit-learn's answer to LightGBM, with no extra dependency.

### The hyperparameters

| Parameter | Default here | Chosen by Optuna | What it does |
|---|---|---|---|
| `max_iter` | 400 | 450 | Number of trees. More = more capacity, more overfitting risk |
| `learning_rate` | 0.06 | 0.0886 | How much each tree contributes. Lower = slower, usually better |
| `max_leaf_nodes` | 31 | 41 | Tree size, i.e. how complex an interaction one tree can express |
| `min_samples_leaf` | 40 | 67 | Minimum rows per leaf. Higher = smoother, less overfitting |
| `l2_regularization` | 1.0 | 8.136 | Shrinks leaf values, like ridge's `alpha` |
| `early_stopping` | **False** | — | See below |

**`early_stopping=False` is a deliberate and important choice.** With it enabled,
scikit-learn carves out a *random* validation slice to decide when to stop. On
this dataset a random slice leaks — it would contain rows from dates adjacent to
the ones being predicted. Disabling it means tree count is governed entirely by
`max_iter`, chosen against the honest forward-in-time folds. This is a small line
with real methodological weight.

### Why the tree is the weakest family here

Measured on the Sep–Oct holdout: **MAE $116.90, MAPE 4.88%, bias 0.951.**

That bias figure is the tell. The tree is systematically **4.9% low** — precisely
the extrapolation failure predicted in [§2.2](#22-the-critical-asymmetry-time).
It has no time feature, so it cannot follow the upward drift and effectively
predicts October prices for December.

The tree is retained because:
1. It is the honest baseline that *demonstrates* the extrapolation problem.
2. It is the residual engine inside the hybrid, where it is not asked to carry
   the trend.

---

## 5. HybridRateModel

The best-performing family, and the one Optuna selected.

```python
class HybridRateModel(RateModel):
    def __init__(self, alpha=1.0, n_centers=..., residual_weight=1.0, **tree_params):
        self.residual_weight = residual_weight
        self.linear = RidgeRateModel(alpha=alpha, n_centers=n_centers)
        self.tree   = TreeRateModel(**tree_params)

    def fit(self, frame):
        self.linear.fit(frame)
        residual = log_rate_per_mile(frame) - self.linear.predict_log_rpm(frame)

        self.tree.features = TreeFeatures(frame["date"].min())
        self._residual_model = HistGradientBoostingRegressor(
            categorical_features=self.tree.features.categorical_mask, **self.tree.params
        )
        self._residual_model.fit(self.tree.features.transform(frame), residual)
        return self

    def predict_log_rpm(self, frame):
        correction = self._residual_model.predict(self.tree.features.transform(frame))
        return self.linear.predict_log_rpm(frame) + self.residual_weight * correction
```

### The division of labour

```
   ┌─────────────────────────────────────────────────────┐
   │ STAGE 1 — Ridge                                     │
   │   level, time trend, smooth main effects,           │
   │   geography                                         │
   │   → can extrapolate past the training window        │
   └──────────────────────┬──────────────────────────────┘
                          │  residual = truth − stage 1
                          ▼
   ┌─────────────────────────────────────────────────────┐
   │ STAGE 2 — Gradient boosted trees                    │
   │   whatever pattern is LEFT OVER: local              │
   │   interactions a straight line cannot express       │
   │   → no time feature, so cannot break extrapolation  │
   └──────────────────────┬──────────────────────────────┘
                          ▼
      prediction = stage1 + residual_weight × stage2
```

Each component does what it is good at. The ridge handles the smooth structure
**and owns the extrapolation**. The tree hunts local interactions — for example
"reefers on short hauls out of this particular region behave differently" — a
pattern a linear model cannot express without an explicit interaction term.

Critically, **the tree never sees a date feature**, so it cannot sabotage the
extrapolation. It can only adjust based on load characteristics.

### `residual_weight` — the shrinkage hedge

Searched over `0.2` to `1.0`; Optuna selected **0.928**.

This scales down the tree's correction. The rationale in the docstring: this
dataset is close to additive in log space, so if there is genuinely little left
for stage 2 to find, the weight can fall toward zero and nothing is lost. It is a
safety valve — the hybrid can gracefully degrade to "just the ridge" if the
residual stage is not earning its keep.

That the search landed at 0.928 says the residual stage **is** contributing
meaningfully here.

### Why this beats both parents

| Model | MAE | MAPE | bias |
|---|---|---|---|
| ridge alone | $54.59 | 2.34% | 0.983 |
| tree alone | $116.90 | 4.88% | 0.951 |
| **hybrid** | **$51.84** | **2.15%** | **0.981** |

The hybrid inherits the ridge's extrapolation (bias 0.981, close to the ridge's
0.983 and far better than the tree's 0.951) while picking up accuracy from the
residual stage.

### ⚠️ Two honest caveats

**1. In-sample residuals.** Stage 2 is trained on residuals from stage 1's
*training* predictions, which are optimistically small. Stage 2 therefore sees a
slightly easier target than it will face at prediction time. The standard fix is
out-of-fold residuals. In practice `residual_weight` and `l2_regularization`
(selected at 8.136, a high value) absorb much of this. Not a bug, but a known
approximation.

**2. Dead code.** `self.tree = TreeRateModel(**tree_params)` is constructed, but
only its `.params` and `.features` are ever borrowed — `self.tree.model` is never
fitted and stays `None`. The hybrid builds its own `_residual_model`. It works,
but it reads confusingly and would be clearer as a plain params dict.

---

## 6. BlendRateModel and MedianRateModel

### MedianRateModel — the floor to beat

```python
class MedianRateModel(RateModel):
    def fit(self, frame):
        self.value = float(np.median(log_rate_per_mile(frame)))
        return self

    def predict_log_rpm(self, frame):
        return np.full(len(frame), self.value)
```

"Every load costs the median rate per mile, times its distance." It uses *no*
features beyond distance.

Measured: **MAE $203.69, MAPE 9.27%.** Every reported improvement should be read
against this number. The hybrid at 2.15% is **4.3× better** — that ratio is the
honest measure of what the modelling added.

Always have a baseline this dumb. Without one, "2.15% MAPE" is a number with no
meaning.

### BlendRateModel — averaging in log space

```python
class BlendRateModel(RateModel):
    def fit(self, frame):
        self.models = [build(name, **params).fit(frame) for name, params in self.specs]
        return self

    def predict_log_rpm(self, frame):
        return np.mean([m.predict_log_rpm(frame) for m in self.models], axis=0)
```

Averaging happens **in log space**, before exponentiating. That makes the result
the *geometric* mean of the dollar predictions, not the arithmetic mean — the
correct average for multiplicative data.

The stated rationale ([`model.py`](../src/model.py)): under a two-month
distribution shift, the ranking of near-tied configurations is not stable enough
to trust, so average the top few rather than betting on the single winner.

**Measured outcome on the actual run — it was neutral:**

| | MAE | RMSE | MAPE | bias |
|---|---|---|---|---|
| single best | $51.36 | $77.54 | 2.126% | 0.98160 |
| blend of 3 | $51.12 | $77.07 | 2.129% | 0.98192 |

MAE improved 0.46%, MAPE got *worse* by 0.12%, bias essentially unchanged. All
three blended configurations were `hybrid` with near-identical hyperparameters, so
they make nearly the same errors and averaging cancels very little.

> **Key principle:** ensembling reduces **variance** (random error that differs
> between models), not **bias** (systematic error they all share). Three scales
> that each read 2 kg light will average to 2 kg light. Blending cannot fix the
> bias discussed in [§9.3](#93-the-stability-warning).

The selection rule is a bare tie-break ([`train.py`](../src/train.py)):

```python
use_blend = blend_scores["clean_mae"] <= single_scores["clean_mae"]
```

Shipping the blend is defensible as a hedge, but the data does not support
claiming it improved the model.

---

## 7. The `build()` factory

```python
def build(name: str, **params) -> RateModel:
    builders = {
        "median": MedianRateModel,
        "ridge":  RidgeRateModel,
        "tree":   TreeRateModel,
        "hybrid": HybridRateModel,
        "blend":  BlendRateModel,
    }
    if name not in builders:
        raise ValueError(f"unknown model family: {name}")
    return builders[name](**params)
```

Small but structurally important: it lets **the model family itself become a
searchable hyperparameter**. In [`train.py`](../src/train.py):

```python
family = trial.suggest_categorical("family", ["ridge", "tree", "hybrid"])
...
result = evaluate.cross_validate(lambda: models.build(family, **params), train, clean_mask)
```

Optuna searches over *which model to use* and *how to configure it* in one space,
instead of requiring three separate studies.

Note the `lambda`. `cross_validate` needs a **fresh, unfitted** model for every
fold, so it is handed a *factory*, not an instance. Passing a single instance would
refit the same object four times and leak state between folds.

---

## 8. A worked example, end to end

Take validation row `TE-000001`:

```
Baton Rouge → Mobile, 331.4 mi, Flatbed, 19,958 lb, 2025-11-01, market_index 0.90402
```

### Step 1 — `Cleaner.transform()`

Weight is positive and present; `market_index` is present. All three defect flags
are 0, values unchanged.

### Step 2 — `base_frame()`

```
log_distance     = ln(331.4)            = 5.8033
log_weight       = ln(19,958)           = 9.9014
market_index     =                        0.9040
haversine        =                      275.3805
circuity         = 331.4 / 275.38       = 1.2034   (20% longer than straight line)
dx               = 4.4018                          (heading east)
dy               = 1.3031                          (heading north)
day_of_week      = 5 (Saturday)  →  is_weekend = 1
days_since_start = 304 / 365            = 0.8329
equipment        = Flatbed
```

Note `days_since_start` is measured from the **training** origin (2025-01-01),
stored on the fitted `LinearFeatures` object — not from the start of whatever
frame is being predicted. This is what makes the trend extrapolate correctly into
the validation period.

### Step 3 — the two paths diverge

**Ridge path** → a 112-column row:

```
  8 linear terms   [5.8033, 0.9040, 9.9014, 1.2034, 0, 0, 0, 0.8329]
 16 hinges         log_distance − 5.788 = 0.015  (only just past the first knot)
                   log_distance − 6.293 = 0      (below, contributes nothing)
                   ... 14 more
  2 equipment      Flatbed = 1, Reefer = 0
  6 day-of-week    dow_5 = 1, rest 0
 80 geo bases      exp(−dist²/2σ²) to each of 40 centres, for both ends
        ↓ StandardScaler  (each column centred and scaled)
        ↓ Ridge.predict
```

**Tree path** → a 17-column row:

```
  [5.8033, 0.9040, 9.9014, 30.501, −92.654, 31.804, −88.252, 275.38, 1.2034,
   4.4018, 1.3031, 5, 1, 0, 0, 0, 1]
                                          ↑ equipment_code for Flatbed
        ↓ 450 trees, each voting a small adjustment
```

**Hybrid** runs both and adds `residual_weight × correction`.

### Step 4 — back to dollars

Actual output of the shipped model:

```python
log_rpm = 0.9536
rate    = np.exp(0.9536) × 331.4 = $859.97      # i.e. $2.595 per mile
```

**Sanity check:** the mean actual rate for comparable training loads (280–390 mi,
Flatbed, n=606) is **$2.590/mile**. The model predicts **$2.595/mile** — within
0.2%. The pipeline is behaving sensibly on this row.

Across all 12,000 validation rows the shipped model produces predictions spanning
**$195.46 to $6,854.92**, mean **$2,384.28**, all positive and finite.

---

## 9. Measured results

### 9.1 The primary forward holdout (fit Jan–Aug → score Sep–Oct)

From the actual 60-trial run:

| Model | MAE | RMSE | MAPE | MedAPE | bias |
|---|---|---|---|---|---|
| median | $203.69 | $295.92 | 9.27% | 7.90% | 1.0305 |
| tree | $116.90 | $149.87 | 4.88% | 4.71% | 0.9506 |
| ridge | $54.59 | $80.27 | 2.34% | 1.96% | 0.9826 |
| **hybrid** | **$51.84** | **$78.03** | **2.15%** | **1.79%** | **0.9814** |

All four metrics rank the families identically — `hybrid > ridge > tree > median`
— so the choice of metric does not change the conclusion. This happens because the
log target makes errors *proportional*, keeping MAPE flat at 1.39–1.44% across
every distance band.

### 9.2 Optuna's verdict

```
best per family:  hybrid=$54.07   ridge=$56.46   tree=$80.92
best params: family=hybrid, alpha=861.7, n_centers=40, learning_rate=0.0886,
             max_leaf_nodes=41, min_samples_leaf=67, l2_regularization=8.136,
             max_iter=450, residual_weight=0.928
```

The hybrid wins cleanly. Note how **high the regularisation settled**: `alpha` at
861.7 out of a `1e-2..1e3` range, `l2_regularization` at 8.136 out of `1e-3..10`.
Both near their ceilings. The search is saying: *stay smooth, do not chase
detail, you are forecasting past the edge of your data.*

### 9.3 The stability warning

The single most important caveat about all three models.

Bias across the four expanding-window folds:

| Fold | MAE | bias | |
|---|---|---|---|
| Jun | $53.39 | 0.98344 | 1.7% **low** |
| Jul | $81.00 | 1.03322 | 3.3% **high** |
| Aug | $33.34 | 1.00762 | 0.8% high |
| Sep–Oct | $51.12 | 0.98192 | 1.8% **low** |

**The bias flips sign.** It is not a fixed offset, so a constant calibration
factor cannot fix it (tested — it made things worse).

The cause is the `days_since_start` slope. Actual rate per mile by month:

```
Jan  Feb  Mar  Apr  May  Jun   Jul  Aug  Sep  Oct
2.07 2.10 2.18 2.19 2.24 2.30  2.23 2.16 2.20 2.21
                          ↑ peak
```

Train through June and the fitted slope projects continued rise into July →
**+3.3% overshoot**. Train through August, after the Jul–Aug dip drags the slope
down → **−1.8% undershoot** on Sep–Oct. The mechanism fully explains the
observed pattern.

**Practical consequence:** expect roughly **2.1% MAPE on the submission, with a
level uncertainty of about ±2%.** That is the honest confidence interval. The
`days_since_start` term is simultaneously the most valuable feature in the model
(removing it doubles the error) and the least stable.

---

## 10. Summary

| Model | Core idea | Can extrapolate? | Measured MAPE | Role |
|---|---|---|---|---|
| `MedianRateModel` | One constant rate/mile | — | 9.27% | Baseline floor |
| `TreeRateModel` | 400 boosted trees, no date | ❌ No | 4.88% | Residual engine; demonstrates the extrapolation problem |
| `RidgeRateModel` | Splines + geo basis + linear trend | ✅ Yes | 2.34% | Carries level and trend |
| `HybridRateModel` | Ridge + tree on the residual | ✅ Yes | **2.15%** | **Shipped** |
| `BlendRateModel` | Geometric mean of top-3 configs | ✅ Yes | 2.13% | Hedge; neutral in measurement |

### The five ideas worth carrying to another project

1. **Model the right target.** `log(rate ÷ mile)` removes the dominant driver and
   linearises the multiplicative structure. Almost all of the accuracy comes from
   this one choice, before any model is fitted.
2. **Trees cannot extrapolate.** If your test set lies beyond your training range
   in time, a pure tree model will silently predict a constant. Something linear
   must carry the trend.
3. **Feed different models different features.** Splines and scaling are essential
   for the ridge and pointless for the tree. One shared matrix would have
   compromised both.
4. **Encode geography as coordinates, never as identity.** Radial bases handle
   unseen cities gracefully; one-hot encoding cannot represent them at all.
5. **Watch bias, not just error.** MAE and MAPE will not tell you the model is
   systematically 2% low. For a forecast that extrapolates past the data, bias is
   the number most likely to expose a broken trend.
