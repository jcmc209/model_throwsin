"""
Train Quantile — Throw-In Predictor
====================================
Entrena tres modelos LightGBM con objective="quantile" (Q25, Q50, Q75) usando
las mismas 30 SHAP features y el mismo split temporal que el modelo principal.

Los modelos cuantil complementan a model_v1.joblib (Poisson) para producir
intervalos de confianza por partido:
  - total_Q25 = home_Q25 + away_Q25  (si > línea → señal OVER con 75% conf.)
  - total_Q75 = home_Q75 + away_Q75  (si < línea → señal UNDER con 75% conf.)
  - Entre Q25 y Q75 → no apostar (incertidumbre alta)

No sobrescribe model_v1.joblib — cambio 100% aditivo.

Uso:
  python -m model.train_quantile
"""
from __future__ import annotations

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
from sklearn.metrics import mean_absolute_error

from model.features import SHAP_SELECTED_FEATURES, TARGET_COL

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
log = logging.getLogger("train_quantile")

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

QUANTILES: list[float] = [0.25, 0.50, 0.75]

CONFIG = {
    "dataset_path": "data/model/dataset.parquet",
    "train_seasons": ["2021/2022", "2022/2023", "2023/2024"],
    "val_seasons": ["2024/2025"],
    "model_paths": {
        0.25: "data/model/model_q25.joblib",
        0.50: "data/model/model_q50.joblib",
        0.75: "data/model/model_q75.joblib",
    },
    "metrics_path": "data/model/metrics_quantile.json",
    # Líneas típicas para análisis (total partido = home + away, media ~40)
    "eval_lines": [35.5, 37.5, 39.5, 41.5, 43.5, 45.5],
    "lgb_params_base": {
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 50,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "verbose": -1,
        "n_estimators": 2000,
        "random_state": 42,
    },
    "early_stopping_rounds": 100,
    "season_decay_weights": {
        "2021/2022": 0.6,
        "2022/2023": 0.8,
        "2023/2024": 1.0,
    },
}

# ─────────────────────────────────────────────────────────────
# CARGA + SPLIT
# ─────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG["dataset_path"])
    log.info("Dataset cargado: %s", df.shape)
    return df


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(CONFIG["train_seasons"])].copy()
    val = df[df["season"].isin(CONFIG["val_seasons"])].copy()
    log.info("Split — train=%d, val=%d", len(train), len(val))
    return train, val


def build_decay_weights(train: pd.DataFrame) -> np.ndarray:
    return train["season"].map(CONFIG["season_decay_weights"]).fillna(1.0).to_numpy()


# ─────────────────────────────────────────────────────────────
# ENTRENAMIENTO CUANTIL
# ─────────────────────────────────────────────────────────────

def train_quantile_model(
    alpha: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray,
) -> lgb.LGBMRegressor:
    params = dict(CONFIG["lgb_params_base"])
    params["objective"] = "quantile"
    params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(CONFIG["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


# ─────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss: métrica estándar para evaluar cuantiles."""
    errors = y_true - y_pred
    return float(np.mean(np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)))


def pinball_baseline(y_true: np.ndarray, alpha: float) -> float:
    """Pinball loss del baseline: predecir siempre la media del train."""
    mean_val = float(np.mean(y_true))
    return pinball_loss(y_true, np.full_like(y_true, mean_val, dtype=float), alpha)


def compute_calibration(
    val: pd.DataFrame,
    preds: dict[float, np.ndarray],
    feature_cols: list[str],
) -> dict:
    """
    Calcula:
    - Cobertura empírica: % de casos donde real está entre Q25 y Q75 (esperado ~50%)
    - Accuracy en líneas O/U a nivel partido (total = home + away)
    - Comparación con Poisson analítico (usando Q50 como estimador puntual)
    """
    q25_pred = preds[0.25]
    q50_pred = preds[0.50]
    q75_pred = preds[0.75]
    y_true = val[TARGET_COL].values

    # Cobertura individual (por equipo)
    coverage_individual = float(np.mean((y_true >= q25_pred) & (y_true <= q75_pred)))

    # MAE del Q50 vs modelo Poisson (comparación)
    mae_q50 = float(mean_absolute_error(y_true, q50_pred))

    # Métricas a nivel partido (home + away sumados)
    home_mask = val["is_home"] == 1
    away_mask = val["is_home"] == 0

    home = val[home_mask].copy()
    away = val[away_mask].copy()

    # Alinear por match_id (misma posición garantizada por sort de train)
    home_sorted = home.sort_values("match_id")
    away_sorted = away.sort_values("match_id")

    if len(home_sorted) != len(away_sorted):
        log.warning("home/away filas desiguales en val; saltando métricas de partido")
        return {
            "coverage_individual_q25_q75": round(coverage_individual, 4),
            "mae_q50_per_team": round(mae_q50, 4),
        }

    # Índices alineados
    h_idx = home_mask.values
    a_idx = away_mask.values

    q25_home = q25_pred[h_idx]
    q25_away = q25_pred[a_idx]
    q50_home = q50_pred[h_idx]
    q50_away = q50_pred[a_idx]
    q75_home = q75_pred[h_idx]
    q75_away = q75_pred[a_idx]

    total_actual = home_sorted[TARGET_COL].values + away_sorted[TARGET_COL].values
    total_q25 = q25_home + q25_away
    total_q50 = q50_home + q50_away
    total_q75 = q75_home + q75_away

    # Clip anti-cruce de cuantiles
    total_q25 = np.minimum(total_q25, total_q50)
    total_q75 = np.maximum(total_q75, total_q50)

    coverage_match = float(np.mean((total_actual >= total_q25) & (total_actual <= total_q75)))
    mae_q50_match = float(mean_absolute_error(total_actual, total_q50))

    # Accuracy en líneas O/U (total partido)
    lines_accuracy: dict[str, dict] = {}
    for line in CONFIG["eval_lines"]:
        model_pred_over = total_q50 > line
        actual_over = total_actual > line
        acc = float(np.mean(model_pred_over == actual_over))
        base_over_rate = float(np.mean(actual_over))

        # Cuando Q25 > línea (señal OVER fuerte)
        strong_over_mask = total_q25 > line
        n_strong_over = int(strong_over_mask.sum())
        acc_strong_over = (
            float(np.mean(actual_over[strong_over_mask]))
            if n_strong_over > 0 else None
        )

        # Cuando Q75 < línea (señal UNDER fuerte)
        strong_under_mask = total_q75 < line
        n_strong_under = int(strong_under_mask.sum())
        acc_strong_under = (
            float(1 - np.mean(actual_over[strong_under_mask]))
            if n_strong_under > 0 else None
        )

        lines_accuracy[str(line)] = {
            "model_acc_q50": round(acc, 4),
            "base_over_rate": round(base_over_rate, 4),
            "advantage_pp": round((acc - max(base_over_rate, 1 - base_over_rate)) * 100, 2),
            "strong_over_n": n_strong_over,
            "strong_over_hit_rate": round(acc_strong_over, 4) if acc_strong_over is not None else None,
            "strong_under_n": n_strong_under,
            "strong_under_hit_rate": round(acc_strong_under, 4) if acc_strong_under is not None else None,
        }
        log.info(
            "  O/U %.1f: acc=%.3f | base=%.3f | Q25>line=%d casos (hit=%.3f) | Q75<line=%d casos (hit=%.3f)",
            line, acc, base_over_rate,
            n_strong_over, acc_strong_over if acc_strong_over else 0,
            n_strong_under, acc_strong_under if acc_strong_under else 0,
        )

    return {
        "coverage_individual_q25_q75": round(coverage_individual, 4),
        "coverage_match_q25_q75": round(coverage_match, 4),
        "mae_q50_per_team": round(mae_q50, 4),
        "mae_q50_match": round(mae_q50_match, 4),
        "lines_accuracy": lines_accuracy,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    df = load_dataset()
    train, val = split_temporal(df)

    missing_feats = [f for f in SHAP_SELECTED_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"SHAP_SELECTED_FEATURES contiene columnas ausentes: {missing_feats}")

    feature_cols = SHAP_SELECTED_FEATURES
    log.info("Features: %d (SHAP_SELECTED_FEATURES)", len(feature_cols))

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_val = val[feature_cols]
    y_val = val[TARGET_COL]
    sw = build_decay_weights(train)

    models: dict[float, lgb.LGBMRegressor] = {}
    preds: dict[float, np.ndarray] = {}
    per_quantile_metrics: dict[str, dict] = {}

    for alpha in QUANTILES:
        label = f"Q{int(alpha * 100)}"
        log.info("Entrenando LightGBM quantile %s (alpha=%.2f) ...", label, alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = train_quantile_model(alpha, X_train, y_train, X_val, y_val, sw)
        models[alpha] = model

        pred = model.predict(X_val)
        preds[alpha] = pred

        pb = pinball_loss(y_val.values, pred, alpha)
        pb_base = pinball_baseline(y_train.values, alpha)
        best_iter = int(model.best_iteration_ or model.n_estimators)

        log.info(
            "  %s → pinball=%.4f | baseline_pinball=%.4f | best_iter=%d",
            label, pb, pb_base, best_iter,
        )
        per_quantile_metrics[label] = {
            "alpha": alpha,
            "pinball_loss": round(pb, 4),
            "pinball_loss_baseline": round(pb_base, 4),
            "beats_baseline": bool(pb < pb_base),
            "best_iteration": best_iter,
        }

    log.info("Calculando métricas de calibración ...")
    calibration = compute_calibration(val, preds, feature_cols)

    # Guardar modelos
    out_dir = Path(CONFIG["model_paths"][0.25]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for alpha in QUANTILES:
        label = f"Q{int(alpha * 100)}"
        path = CONFIG["model_paths"][alpha]
        artifact = {
            "model": models[alpha],
            "alpha": alpha,
            "features": feature_cols,
            "train_seasons": CONFIG["train_seasons"],
            "val_seasons": CONFIG["val_seasons"],
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "metrics": per_quantile_metrics[label],
        }
        joblib.dump(artifact, path)
        log.info("Modelo %s guardado: %s", label, path)

    # Guardar métricas
    metrics_out = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "features_mode": "shap30",
        "n_features": len(feature_cols),
        "train_seasons": CONFIG["train_seasons"],
        "val_seasons": CONFIG["val_seasons"],
        "quantile_metrics": per_quantile_metrics,
        "calibration": calibration,
        "interpretation": {
            "coverage_expected": 0.50,
            "coverage_actual_match": calibration.get("coverage_match_q25_q75"),
            "note": (
                "Si coverage < 0.40: intervalos demasiado estrechos (subconfianza). "
                "Si coverage > 0.65: intervalos demasiado anchos (sobreconfianza). "
                "Objetivo: 0.40-0.60."
            ),
        },
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
    log.info("Métricas cuantil guardadas: %s", CONFIG["metrics_path"])

    # Resumen final
    cov = calibration.get("coverage_match_q25_q75", calibration.get("coverage_individual_q25_q75"))
    log.info(
        "RESUMEN: cobertura Q25-Q75 (partido) = %.1f%% | MAE Q50 partido = %.4f",
        (cov or 0) * 100,
        calibration.get("mae_q50_match", calibration.get("mae_q50_per_team", 0)),
    )
    log.info(
        "ESTRATEGIA: Q25_total > línea → OVER | Q75_total < línea → UNDER | entre Q25-Q75 → no apostar"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
