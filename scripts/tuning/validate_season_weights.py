"""
Validate season-weighted-training change via bootstrap CV comparison.

Compares two weight schemes on all 5 walk-forward folds:
  A (baseline) : auto-decay weights (current _build_decay_weights behavior)
  B (candidate): steeper custom decay weights from the proposal

Bootstrap result tells us whether the new weight scheme is significantly
better before we overwrite model_v1.joblib.
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Make sure we can import from model/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.train import (
    CONFIG,
    load_dataset,
    train_lightgbm,
    compare_experiments_bootstrap,
    _build_decay_weights,
    _decay_sample_weights,
)
from model.features import SHAP_SELECTED_FEATURES, TARGET_COL

# ────────────────────────────────────────────────
# STEEPER WEIGHTS for the candidate experiment
# ────────────────────────────────────────────────
CANDIDATE_WEIGHTS: dict[str, float] = {
    "2019/2020": 0.1,
    "2020/2021": 0.2,
    "2021/2022": 0.2,
    "2022/2023": 0.4,
    "2023/2024": 0.7,
    "2024/2025": 1.0,
}


def _run_folds(df: pd.DataFrame, weight_fn, label: str) -> dict:
    """
    Run all 5 CV folds and collect per-fold MAE + pooled predictions.

    weight_fn(train_seasons) -> dict[season, float]
    """
    feature_cols = SHAP_SELECTED_FEATURES
    folds_out = []
    pred_frames = []

    for fold_cfg in CONFIG["cv_folds"]:
        name = fold_cfg["name"]
        train_seasons = fold_cfg["train_seasons"]
        val_season = fold_cfg["val_season"]
        is_partial = fold_cfg["is_partial"]

        train = df[df["season"].isin(train_seasons)].copy()
        val = df[df["season"] == val_season].copy()

        if train.empty or val.empty:
            print(f"  [SKIP] {name}: train={len(train)}, val={len(val)}")
            continue

        decay_map = weight_fn(train_seasons)
        sw = _decay_sample_weights(train, decay_map)

        model = train_lightgbm(
            train[feature_cols], train[TARGET_COL],
            val[feature_cols], val[TARGET_COL],
            sw,
        )
        preds = model.predict(val[feature_cols])
        val_mae = float(mean_absolute_error(val[TARGET_COL], preds))
        print(f"  {name}: val_MAE={val_mae:.4f}{'  [PARTIAL]' if is_partial else ''} "
              f"  weights={dict(list(decay_map.items())[-3:])}")

        folds_out.append({
            "name": name, "val_season": val_season,
            "is_partial": is_partial, "val_mae": round(val_mae, 4),
        })

        if not is_partial:
            pred_frames.append(pd.DataFrame({
                "match_id": val["match_id"].values,
                "is_home": val["is_home"].values,
                "y_true": val[TARGET_COL].values,
                "y_pred": preds,
                "fold": name,
            }))

    complete = [f for f in folds_out if not f["is_partial"]]
    maes = [f["val_mae"] for f in complete]
    mean_mae = float(np.mean(maes)) if maes else 0.0
    std_mae = float(np.std(maes, ddof=0)) if maes else 0.0
    predictions_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    print(f"  → {label}: mean_val_MAE={mean_mae:.4f} | std={std_mae:.4f}")
    return {
        "label": label,
        "n_features": len(feature_cols),
        "n_complete_folds": len(complete),
        "mean_val_mae": round(mean_mae, 4),
        "std_val_mae": round(std_mae, 4),
        "folds": folds_out,
        "predictions": predictions_df,
    }


def candidate_weight_fn(train_seasons: list[str]) -> dict[str, float]:
    """Custom steeper weights — seasons not in CANDIDATE_WEIGHTS fall back to auto."""
    auto = _build_decay_weights(train_seasons)
    # Override with our steeper values where applicable
    result = {}
    for s in train_seasons:
        result[s] = CANDIDATE_WEIGHTS.get(s, auto.get(s, 0.2))
    return result


def main() -> None:
    print("=" * 60)
    print("BOOTSTRAP VALIDATION: season-weighted-training")
    print("=" * 60)

    df = load_dataset()

    print("\n--- Baseline (auto decay weights) ---")
    result_a = _run_folds(df, _build_decay_weights, label="baseline_auto")

    print("\n--- Candidate (steeper decay weights) ---")
    result_b = _run_folds(df, candidate_weight_fn, label="candidate_steeper")

    print("\n--- Bootstrap comparison ---")
    cmp = compare_experiments_bootstrap(result_a, result_b, n_boot=2000)

    delta = cmp["delta_mean"]
    ci_low = cmp["delta_ci_low"]
    ci_high = cmp["delta_ci_high"]
    p = cmp["p_value"]
    sig = cmp["significant"]

    print(f"\n  Δ MAE (candidate − baseline) = {delta:+.4f}")
    print(f"  IC 95% = [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  p = {p:.3f}  |  significant = {sig}")

    if delta < 0 and sig:
        verdict = "✓ MEJORA SIGNIFICATIVA — proceder con el retrain"
    elif delta < 0 and not sig:
        verdict = "~ Mejora no significativa — retrain con precaución (dirección correcta)"
    else:
        verdict = "✗ No mejora — revisar enfoque antes de reentrenar"

    print(f"\n  VEREDICTO: {verdict}")

    # Save results
    out = {
        "change": "season-weighted-training",
        "baseline": {k: v for k, v in result_a.items() if k != "predictions"},
        "candidate": {k: v for k, v in result_b.items() if k != "predictions"},
        "bootstrap": cmp,
        "candidate_weights": CANDIDATE_WEIGHTS,
        "verdict": verdict,
    }
    out_path = Path("data/model/season_weights_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_native(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"No serializable: {type(obj)}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_to_native)
    print(f"\n  Resultados guardados: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
