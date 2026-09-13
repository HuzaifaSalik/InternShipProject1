"""Model selection and final fit.

Run as:  python -m src.train [--trials N]

The Optuna objective is the weighted mean MAE across expanding-window forward
folds. That choice matters more than the search itself: random K-fold leaks on
this dataset, and a search pointed at a leaky objective will spend every trial
optimising into the leak while reporting a record score.
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import optuna

from . import config, data, evaluate, model as models

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial: optuna.Trial) -> tuple[str, dict]:
    """Search over the model family itself, not just its hyperparameters."""
    family = trial.suggest_categorical("family", ["ridge", "tree", "hybrid"])

    linear = {
        "alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True),
        "n_centers": trial.suggest_int("n_centers", 12, 40, step=4),
    }
    tree = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 200, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_iter": trial.suggest_int("max_iter", 150, 600, step=50),
    }

    if family == "ridge":
        return family, linear
    if family == "tree":
        return family, tree
    return family, {
        **linear,
        **tree,
        "residual_weight": trial.suggest_float("residual_weight", 0.2, 1.0),
    }


def run_study(train, clean_mask, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        family, params = suggest(trial)
        result = evaluate.cross_validate(
            lambda: models.build(family, **params), train, clean_mask
        )
        for fold in result["folds"]:
            trial.set_user_attr(fold["fold"], round(fold["clean_mae"], 3))
        return result["objective_mae"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
        study_name="freight-rate",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def top_specs(study: optuna.Study, k: int) -> list[tuple[str, dict]]:
    """The k best completed trials, rebuilt as (family, params) pairs."""
    done = [t for t in study.trials if t.value is not None]
    specs = []
    for trial in sorted(done, key=lambda t: t.value)[:k]:
        fixed = optuna.trial.FixedTrial(trial.params)
        specs.append(suggest(fixed))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and fit the freight rate model.")
    parser.add_argument("--trials", type=int, default=60, help="Optuna trials")
    parser.add_argument("--blend-top", type=int, default=3, help="configs to average")
    args = parser.parse_args()

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    train, cleaner, outliers = data.load_clean_train()
    clean_mask = ~outliers
    print(f"Loaded {len(train):,} training rows "
          f"({train.date.min().date()} .. {train.date.max().date()})")
    print(f"Corrupted labels excluded from training: {int(outliers.sum()):,} "
          f"({outliers.mean() * 100:.2f}%)\n")

    report: dict = {
        "n_train": int(len(train)),
        "n_labels_excluded": int(outliers.sum()),
    }

    # --- baselines on the primary forward holdout ----------------------------
    tr_index, te_index = evaluate.primary_holdout(train)
    print(f"Primary holdout  fit {train.loc[tr_index].date.min().date()}"
          f"..{train.loc[tr_index].date.max().date()}"
          f"  ->  score {train.loc[te_index].date.min().date()}"
          f"..{train.loc[te_index].date.max().date()}")
    print(f"{'model':>10s} {'MAE':>10s} {'RMSE':>10s} {'MAPE':>8s} {'MedAPE':>8s} {'bias':>8s}")
    baselines = {}
    for name in ("median", "ridge", "tree", "hybrid"):
        scores = evaluate.score_split(
            lambda n=name: models.build(n), train, tr_index, te_index, clean_mask
        )
        baselines[name] = scores
        print(f"{name:>10s} {scores['clean_mae']:10.2f} {scores['clean_rmse']:10.2f} "
              f"{scores['clean_mape']:7.2f}% {scores['clean_median_ape']:7.2f}% "
              f"{scores['clean_bias']:8.4f}")
    report["holdout_baselines"] = baselines

    # --- Optuna over the model zoo -------------------------------------------
    print(f"\nOptuna: {args.trials} trials against the forward-fold objective ...")
    started = time.time()
    study = run_study(train, clean_mask, args.trials)
    print(f"  best objective MAE ${study.best_value:.2f}  "
          f"({time.time() - started:.0f}s, {len(study.trials)} trials)")
    print(f"  best params: {study.best_params}")

    families = {}
    for trial in study.trials:
        if trial.value is not None:
            families.setdefault(trial.params["family"], []).append(trial.value)
    print("  best per family: " + "  ".join(
        f"{k}=${min(v):.2f}" for k, v in sorted(families.items())))

    report["optuna"] = {
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_per_family": {k: min(v) for k, v in families.items()},
    }

    # --- blend the top configurations ----------------------------------------
    specs = top_specs(study, args.blend_top)
    print(f"\nBlending top {len(specs)} configurations: "
          f"{[name for name, _ in specs]}")
    blend_scores = evaluate.score_split(
        lambda: models.BlendRateModel(specs), train, tr_index, te_index, clean_mask
    )
    single_family, single_params = specs[0]
    single_scores = evaluate.score_split(
        lambda: models.build(single_family, **single_params),
        train, tr_index, te_index, clean_mask,
    )
    print(f"  single best  MAE ${single_scores['clean_mae']:.2f}  "
          f"MAPE {single_scores['clean_mape']:.2f}%")
    print(f"  blend        MAE ${blend_scores['clean_mae']:.2f}  "
          f"MAPE {blend_scores['clean_mape']:.2f}%")

    use_blend = blend_scores["clean_mae"] <= single_scores["clean_mae"]
    chosen = "blend" if use_blend else "single"
    print(f"  -> shipping the {chosen}")
    report["selection"] = {
        "chosen": chosen,
        "specs": [[name, params] for name, params in specs],
        "blend_holdout": blend_scores,
        "single_holdout": single_scores,
    }

    def build_final():
        return models.BlendRateModel(specs) if use_blend else models.build(
            single_family, **single_params
        )

    # --- leakage contrast: what random K-fold would have told us -------------
    print("\nLeakage check (same model, two validation designs):")
    forward = evaluate.cross_validate(build_final, train, clean_mask)
    random_scores = [
        evaluate.score_split(build_final, train, tr, te, clean_mask)["clean_mae"]
        for tr, te, _ in evaluate.random_folds(train, n_splits=4)
    ]
    forward_final = forward["folds"][-1]["clean_mae"]
    random_mean = float(np.mean(random_scores))
    print(f"  random K-fold        MAE ${random_mean:7.2f}   <- optimistic, not used")
    print(f"  forward Sep-Oct fold MAE ${forward_final:7.2f}   <- honest estimate")
    print(f"  random K-fold understates error by "
          f"{(1 - random_mean / forward_final) * 100:.1f}%")
    report["leakage_check"] = {
        "random_kfold_mae": random_mean,
        "forward_final_fold_mae": forward_final,
    }
    report["forward_cv"] = forward

    # --- refit on every clean training row -----------------------------------
    print("\nRefitting on all clean training rows ...")
    final = build_final().fit(train.loc[clean_mask])
    joblib.dump({"model": final, "cleaner": cleaner}, config.MODEL_PKL)
    print(f"  saved {config.MODEL_PKL.relative_to(config.ROOT)}")

    config.METRICS_JSON.write_text(json.dumps(report, indent=2, default=float))
    print(f"  saved {config.METRICS_JSON.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
