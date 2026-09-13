"""Generate the two submission artefacts from a fitted model.

Run as:  python -m src.predict

Reads `outputs/model.joblib` (written by `python -m src.train`) and produces:

1. `validation_predictions.csv` -- load_id,predicted_rate for all 12,000 rows of
   data/validation.csv, in the order given by the supplied template.
2. `data/december_chart_inputs.csv` -- the same seven columns it arrived with,
   with `predicted_rate` filled in.

The December file is the awkward one. It carries only pickup, delivery,
distance, equipment, weight and date, but the model also needs coordinates and
`market_index`. Both are recovered rather than invented:

* Coordinates: every city maps to exactly one lat/lon pair throughout the
  dataset, so Lexington and Fort Wayne are looked up from the labelled data.
* market_index: a property of the day, not of the load (within-date std 0.025).
  validation.csv contains ~199 real loads for every December day, so the daily
  mean of those is the market condition for that date -- measured, not assumed.

Neither step touches `posted_rate`, so neither is leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data

DECEMBER_COLUMNS = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "date",
    "predicted_rate",
]


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon pair per city, gathered from both ends of every haul."""
    parts = []
    for frame in frames:
        for role in ("pickup", "delivery"):
            parts.append(
                frame[[role, f"{role}_lat", f"{role}_lon"]].rename(
                    columns={role: "city", f"{role}_lat": "lat", f"{role}_lon": "lon"}
                )
            )
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def predict_validation(model, cleaner) -> pd.DataFrame:
    """Predictions for validation.csv, ordered to match the supplied template."""
    frame = cleaner.transform(data.load_raw(config.VALID_CSV))
    predicted = pd.Series(model.predict_rate(frame), index=frame["load_id"].to_numpy())

    template = pd.read_csv(config.TEMPLATE_CSV)
    missing = set(template["load_id"]) - set(predicted.index)
    if missing:
        raise SystemExit(f"no prediction for {len(missing)} template load_id values")

    return pd.DataFrame(
        {
            "load_id": template["load_id"],
            "predicted_rate": predicted.loc[template["load_id"]].to_numpy().round(2),
        }
    )


def predict_december(model, cleaner, train: pd.DataFrame) -> pd.DataFrame:
    """Fill the December scenario, recovering the columns the file omits."""
    validation = cleaner.transform(data.load_raw(config.VALID_CSV))
    signals = data.daily_market_signals(validation)
    coordinates = city_coordinates(train, validation)

    frame = data.load_raw(config.DECEMBER_CSV)
    for role, city in (("pickup", config.DEC_PICKUP), ("delivery", config.DEC_DELIVERY)):
        if city not in coordinates.index:
            raise SystemExit(f"no coordinates known for {city}")
        frame[f"{role}_lat"] = coordinates.loc[city, "lat"]
        frame[f"{role}_lon"] = coordinates.loc[city, "lon"]

    frame = frame.join(signals[["market_index"]], on="date")
    if frame["market_index"].isna().any():
        # No validation loads on some December day: fall back to the cleaner's
        # global median rather than emitting a NaN the model cannot consume.
        frame["market_index"] = frame["market_index"].fillna(cleaner.market_index_global)

    prepared = cleaner.transform(frame)
    frame["predicted_rate"] = model.predict_rate(prepared).round(2)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame[DECEMBER_COLUMNS]


def main() -> None:
    if not config.MODEL_PKL.is_file():
        raise SystemExit(f"{config.MODEL_PKL} not found -- run `python -m src.train` first")

    import joblib

    bundle = joblib.load(config.MODEL_PKL)
    model, cleaner = bundle["model"], bundle["cleaner"]
    print(f"Loaded {type(model).__name__} from {config.MODEL_PKL.relative_to(config.ROOT)}")

    train = cleaner.transform(data.load_raw(config.TRAIN_CSV))

    predictions = predict_validation(model, cleaner)
    predictions.to_csv(config.PREDICTIONS_CSV, index=False)
    rate = predictions["predicted_rate"]
    print(
        f"  {config.PREDICTIONS_CSV.name}: {len(predictions):,} rows, "
        f"${rate.min():,.2f}..${rate.max():,.2f} (mean ${rate.mean():,.2f})"
    )

    december = predict_december(model, cleaner, train)
    december.to_csv(config.DECEMBER_CSV, index=False)
    dec_rate = december["predicted_rate"]
    print(
        f"  {config.DECEMBER_CSV.name}: {len(december)} rows, "
        f"${dec_rate.min():,.2f}..${dec_rate.max():,.2f} "
        f"(swing {100 * (dec_rate.max() / dec_rate.min() - 1):.1f}%)"
    )

    if (rate <= 0).any() or (dec_rate <= 0).any():
        raise SystemExit("non-positive predictions produced -- score.py would reject these")
    print("\nBoth files written. Now run:")
    print("  python score.py --predictions validation_predictions.csv "
          "--december-predictions data/december_chart_inputs.csv")


if __name__ == "__main__":
    main()
