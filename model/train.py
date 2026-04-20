"""
Train — Throw-In Predictor
==========================
Entrena el modelo predictivo de saques de banda por equipo.

Pipeline:
  1. Carga data/model/dataset.parquet
  2. Split temporal walk-forward:
       train = 2021/2022, 2022/2023, 2023/2024
       val   = 2024/2025
       test  = 2025/2026 (reservado, no se usa hasta que termine la temporada)
  3. Calcula baseline MAE (media por equipo) en val
  4. Entrena:
       - LightGBM Poisson con sample_weight uniforme
       - LightGBM Poisson con sample_weight decay por temporada
       - Negative Binomial (statsmodels) como sanity check
  5. Evalúa en val (MAE, RMSE, MAE por equipo, MAE por temporada)
  6. Selecciona mejor LightGBM (menor MAE val), warning si no bate baseline
  7. Guarda:
       data/model/model_v1.joblib       (modelo + metadata)
       data/model/metrics_v1.json       (métricas de las 3 configuraciones)
       data/model/feature_importance.csv (gain + split ranking)

Uso:
  python -m model.train
  python -m model.train --weights uniform   # solo uniforme
  python -m model.train --weights decay     # solo decay
  python -m model.train --weights both      # ambos (default)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.features import TARGET_COL, get_feature_columns

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("train")

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "dataset_path": "data/model/dataset.parquet",
    "model_path": "data/model/model_v1.joblib",
    "metrics_path": "data/model/metrics_v1.json",
    "feature_importance_path": "data/model/feature_importance.csv",
    "train_seasons": ["2021/2022", "2022/2023", "2023/2024"],
    "val_seasons": ["2024/2025"],
    "test_seasons": ["2025/2026"],
    "baseline_target_mae": 4.84,
    "season_decay_weights": {
        "2021/2022": 0.6,
        "2022/2023": 0.8,
        "2023/2024": 1.0,
    },
    "lgb_params": {
        "objective": "poisson",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 2000,
        "random_state": 42,
    },
    "early_stopping_rounds": 50,
}


# ─────────────────────────────────────────────────────────────
# LOAD + SPLIT
# ─────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG["dataset_path"])
    log.info("Dataset cargado: %s", df.shape)
    return df


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(CONFIG["train_seasons"])].copy()
    val = df[df["season"].isin(CONFIG["val_seasons"])].copy()
    test = df[df["season"].isin(CONFIG["test_seasons"])].copy()

    # Sanity: sin intersección por match_id×is_home
    ids = lambda d: set(zip(d["match_id"], d["is_home"]))
    assert not (ids(train) & ids(val)), "Intersección train/val"
    assert not (ids(train) & ids(test)), "Intersección train/test"
    assert not (ids(val) & ids(test)), "Intersección val/test"

    log.info("Split temporal — train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


# ─────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────

def baseline_team_mean(train: pd.DataFrame, val: pd.DataFrame) -> float:
    """MAE del baseline: predicción = media histórica del equipo en train."""
    team_mean = train.groupby("team_id")[TARGET_COL].mean()
    global_mean = train[TARGET_COL].mean()
    val_pred = val["team_id"].map(team_mean).fillna(global_mean)
    mae = mean_absolute_error(val[TARGET_COL], val_pred)
    log.info("Baseline (media por equipo) — MAE val = %.4f", mae)
    return mae


# ─────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────

def evaluate(y_true: pd.Series, y_pred: np.ndarray, groups: dict[str, pd.Series]) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    out = {"mae": mae, "rmse": rmse}
    for name, grp in groups.items():
        by_group = (
            pd.DataFrame({"y": y_true.values, "p": y_pred, "g": grp.values})
            .groupby("g")
            .apply(lambda d: mean_absolute_error(d["y"], d["p"]))
            .to_dict()
        )
        out[f"mae_by_{name}"] = {str(k): round(float(v), 4) for k, v in by_group.items()}
    return out


# ─────────────────────────────────────────────────────────────
# LIGHTGBM
# ─────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray | None,
) -> lgb.LGBMRegressor:
    params = dict(CONFIG["lgb_params"])
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(CONFIG["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def build_sample_weights(train: pd.DataFrame, scheme: str) -> np.ndarray | None:
    if scheme == "uniform":
        return None
    if scheme == "decay":
        return train["season"].map(CONFIG["season_decay_weights"]).fillna(1.0).to_numpy()
    raise ValueError(f"sample weights scheme desconocido: {scheme}")


# ─────────────────────────────────────────────────────────────
# NEGATIVE BINOMIAL (sanity check)
# ─────────────────────────────────────────────────────────────

def train_negbinom(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    """NegBinom sobre subconjunto reducido de features. Solo para comparar."""
    try:
        import statsmodels.api as sm
    except ImportError:
        log.warning("statsmodels no instalado; saltando NegBinom")
        return {}

    reduced = [
        "is_home",
        "rolling5_throw_ins_total",
        "opp_rolling5_throw_ins_total",
        "rolling10_throw_ins_total",
        "std_throw_ins_total",
    ]
    reduced = [c for c in reduced if c in train.columns]
    X_tr = sm.add_constant(train[reduced].fillna(train[reduced].median()))
    y_tr = train[TARGET_COL]
    X_va = sm.add_constant(val[reduced].fillna(train[reduced].median()), has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y_tr, X_tr, family=sm.families.NegativeBinomial()).fit(maxiter=100)
    except Exception as exc:
        log.warning("NegBinom falló: %s", exc)
        return {}

    pred = model.predict(X_va).to_numpy()
    return evaluate(val[TARGET_COL], pred, {"team": val["team_id"], "season": val["season"]})


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(weights_mode: str = "both") -> None:
    df = load_dataset()
    train, val, test = split_temporal(df)

    baseline_mae = baseline_team_mean(train, val)

    feature_cols = get_feature_columns(df)
    # Excluimos columnas que podrían ser leakage si aparecieran (defensa extra)
    leakage_suspect = [c for c in feature_cols if c in train.columns and c.startswith(
        ("shots_", "passes_accurate", "passes_key", "pass_success", "tackles_",
         "interceptions", "clearances", "dribbles_", "aerials_won", "aerials_offensive",
         "aerials_defensive", "aerial_success_pct", "touches_total", "corners_accurate",
         "throw_ins_accurate", "throw_in_accuracy", "dispossessed", "dribble_success_pct",
         "dribbled_past", "shots_on_post", "tackle_success_pct", "fouls_committed",
         "attendance")
    )]
    # NOTA: columnas base del partido actual (p.ej. corners_total, passes_total) se
    # filtran aquí porque son estadísticas post-partido, NO pre-match.
    raw_match_stats = [
        "throw_ins_accurate", "throw_in_accuracy_pct",
        "shots_total", "shots_on_target", "shots_off_target", "shots_blocked", "shots_on_post",
        "passes_total", "passes_accurate", "passes_key", "pass_success_pct",
        "aerials_total", "aerials_won", "aerials_offensive", "aerials_defensive", "aerial_success_pct",
        "tackles_total", "tackles_successful", "tackles_unsuccessful", "tackle_success_pct",
        "dribbled_past", "dribbles_won", "dribbles_attempted", "dribbles_lost", "dribble_success_pct",
        "interceptions", "clearances", "fouls_committed", "dispossessed",
        "corners_total", "corners_accurate", "touches_total", "possession_pct", "attendance",
    ]
    feature_cols = [c for c in feature_cols if c not in raw_match_stats]
    log.info("Features tras filtrar raw match stats: %d", len(feature_cols))

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_val = val[feature_cols]
    y_val = val[TARGET_COL]

    schemes = ["uniform", "decay"] if weights_mode == "both" else [weights_mode]

    results: dict[str, dict] = {}
    models: dict[str, lgb.LGBMRegressor] = {}
    for scheme in schemes:
        log.info("Entrenando LightGBM Poisson (weights=%s) ...", scheme)
        sw = build_sample_weights(train, scheme)
        model = train_lightgbm(X_train, y_train, X_val, y_val, sw)
        pred = model.predict(X_val)
        metrics = evaluate(y_val, pred, {"team": val["team_id"], "season": val["season"]})
        metrics["best_iteration"] = int(model.best_iteration_ or model.n_estimators)
        log.info("  weights=%s → MAE %.4f | RMSE %.4f | best_iter %d",
                 scheme, metrics["mae"], metrics["rmse"], metrics["best_iteration"])
        results[f"lgbm_{scheme}"] = metrics
        models[scheme] = model

    log.info("Entrenando NegBinom sanity check ...")
    negbinom_metrics = train_negbinom(train, val)
    if negbinom_metrics:
        results["negbinom"] = negbinom_metrics
        log.info("  negbinom → MAE %.4f | RMSE %.4f", negbinom_metrics["mae"], negbinom_metrics["rmse"])

    # Selección del mejor
    best_scheme = min(models, key=lambda s: results[f"lgbm_{s}"]["mae"])
    best_model = models[best_scheme]
    best_mae = results[f"lgbm_{best_scheme}"]["mae"]
    log.info("Mejor configuración: lgbm_%s (MAE %.4f)", best_scheme, best_mae)

    success = best_mae < baseline_mae
    if success:
        log.info("✓ Modelo bate baseline (%.4f < %.4f)", best_mae, baseline_mae)
    else:
        log.warning("✗ Modelo NO bate baseline (%.4f >= %.4f) — iterar features/params",
                    best_mae, baseline_mae)

    # Persistencia
    model_dir = Path(CONFIG["model_path"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_model,
        "version": "v1",
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "features": feature_cols,
        "params": dict(CONFIG["lgb_params"]),
        "val_mae": best_mae,
        "val_rmse": results[f"lgbm_{best_scheme}"]["rmse"],
        "baseline_mae": baseline_mae,
        "baseline_target_mae": CONFIG["baseline_target_mae"],
        "sample_weights_scheme": best_scheme,
        "beats_baseline": success,
        "train_seasons": CONFIG["train_seasons"],
        "val_seasons": CONFIG["val_seasons"],
    }
    joblib.dump(artifact, CONFIG["model_path"])
    log.info("Modelo guardado: %s", CONFIG["model_path"])

    metrics_out = {
        "trained_at": artifact["trained_at"],
        "baseline_mae": baseline_mae,
        "baseline_target_mae": CONFIG["baseline_target_mae"],
        "best_model": f"lgbm_{best_scheme}",
        "beats_baseline": success,
        "n_features": len(feature_cols),
        "train_rows": len(train),
        "val_rows": len(val),
        "results": results,
    }
    def _to_native(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"No serializable: {type(obj).__name__}")

    with open(CONFIG["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False, default=_to_native)
    log.info("Métricas guardadas: %s", CONFIG["metrics_path"])

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": best_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": best_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(CONFIG["feature_importance_path"], index=False)
    log.info("Feature importance guardado: %s", CONFIG["feature_importance_path"])
    log.info("Top 10 features por gain:\n%s", fi.head(10).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train throw-in predictor")
    parser.add_argument(
        "--weights",
        choices=["uniform", "decay", "both"],
        default="both",
        help="Esquema de sample weights a probar",
    )
    args = parser.parse_args()
    main(weights_mode=args.weights)
