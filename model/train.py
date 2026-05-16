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

Guardrail (team bias calibration):
  Cualquier bloque que genere o evalúe `team_bias_calibration_v2.json`
  DEBE importar desde `model.market_utils` (`load_team_bias`, `apply_team_bias`,
  `shrink_bias`). No inlinear lógica de carga/aplicación/shrinkage en este archivo —
  el helper es la fuente única consumida por `model/predict.py`. La regeneración
  del JSON vive en `_regenerate_team_bias_calibration()` más abajo y se invoca
  al final de `main()` tras guardar el joblib.
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model.features import SHAP_SELECTED_FEATURES, TARGET_COL, get_feature_columns
from model.market_utils import shrink_bias

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
    "match_dataset_path": "data/model/dataset_match.parquet",
    "model_path": "data/model/model_v1.joblib",
    "model_total_path": "data/model/model_v1_total.joblib",
    "share_coefs_path": "data/model/share_coefs.json",
    "metrics_path": "data/model/metrics_v1.json",
    "metrics_bivariate_path": "data/model/metrics_bivariate.json",
    "feature_importance_path": "data/model/feature_importance.csv",
    "feature_importance_total_path": "data/model/feature_importance_total.csv",
    "cv_results_path": "data/model/cv_results.json",
    "cv_results_csv_path": "data/model/cv_results.csv",
    "team_bias_path": "data/model/team_bias_calibration_v2.json",
    "team_bias_pre_regen_bak_path": "data/model/team_bias_calibration_v2_pre_regen_bak.json",
    # Shrinkage prior (reverse-engineered del JSON frozen 2026-04-21):
    # k = sigma2_res / sigma2_prior = 5.0, prior_mu = 0.0.
    # Verificado con reverse-fit sobre las 40 filas (std(k)=0.002, rounding-level).
    "team_bias_shrinkage_k": 5.0,
    "team_bias_prior_mu": 0.0,
    "current_per_team_mae": 4.3379,
    "train_seasons": ["2021/2022", "2022/2023", "2023/2024", "2024/2025"],
    "val_seasons": ["2025/2026"],
    "test_seasons": [],
    "cv_folds": [
        {
            "name": "fold0_21-22",
            "train_seasons": ["2019/2020", "2020/2021"],
            "val_season": "2021/2022",
            "is_partial": False,
        },
        {
            "name": "fold1_22-23",
            "train_seasons": ["2019/2020", "2020/2021", "2021/2022"],
            "val_season": "2022/2023",
            "is_partial": False,
        },
        {
            "name": "fold2_23-24",
            "train_seasons": ["2019/2020", "2020/2021", "2021/2022", "2022/2023"],
            "val_season": "2023/2024",
            "is_partial": False,
        },
        {
            "name": "fold3_24-25",
            "train_seasons": ["2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024"],
            "val_season": "2024/2025",
            "is_partial": False,
        },
        {
            "name": "fold4_25-26",
            "train_seasons": ["2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"],
            "val_season": "2025/2026",
            "is_partial": True,
        },
    ],
    "baseline_target_mae": 4.84,
    "season_decay_weights": {
        # Auto-decay pattern for 4 train seasons: most recent=1.0, each older −0.2 (min 0.2)
        # Bootstrap confirmed auto-decay outperforms steeper schemes (see validate_season_weights.py)
        "2019/2020": 0.2,  # used in CV folds only
        "2020/2021": 0.2,  # used in CV folds only
        "2021/2022": 0.4,
        "2022/2023": 0.6,
        "2023/2024": 0.8,
        "2024/2025": 1.0,
    },
    "lgb_params": {
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "metric": "mae",
        # Hiperparámetros optimizados por Optuna (60 trials, 2026-04-21) para Poisson.
        # Tweedie p=1.2 usa los mismos params (corr=0.9924 con Poisson → óptimos transferibles).
        # Exploración 2026-04-21: val_MAE Tweedie=3.7941 vs Poisson=3.8061 (Δ=−0.012).
        # Bootstrap CV (4 folds): Δ=−0.0052, IC95=[−0.0162,+0.0051], p=0.172 (no sig.).
        "learning_rate": 0.049,
        "num_leaves": 4,
        "min_child_samples": 115,
        "feature_fraction": 0.543,
        "bagging_fraction": 0.712,
        "bagging_freq": 5,
        "reg_lambda": 8.10,
        "reg_alpha": 2.83,
        "verbose": -1,
        "n_estimators": 2000,
        "random_state": 42,
    },
    "early_stopping_rounds": 100,
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
    lgb_params_override: dict | None = None,
) -> lgb.LGBMRegressor:
    params = dict(CONFIG["lgb_params"])
    if lgb_params_override:
        params.update(lgb_params_override)
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
    """NegBinom sobre subconjunto reducido de features.

    Devuelve `{"metrics": ..., "model": fitted_glm, "features": [5 cols], "train_medians": {...}}`
    cuando el ajuste es exitoso — el modelo se persiste en el joblib para el ensemble
    (ver `main()`). Si statsmodels no está instalado o el GLM falla, devuelve `{}`.

    Las medianas de train se incluyen para reproducir exactamente la imputación de NaNs
    en tiempo de inferencia (misma semántica que `fillna(train[reduced].median())`).
    """
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
    train_medians = train[reduced].median()
    X_tr = sm.add_constant(train[reduced].fillna(train_medians))
    y_tr = train[TARGET_COL]
    X_va = sm.add_constant(val[reduced].fillna(train_medians), has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.GLM(y_tr, X_tr, family=sm.families.NegativeBinomial()).fit(maxiter=100)
    except Exception as exc:
        log.warning("NegBinom falló: %s", exc)
        return {}

    pred = model.predict(X_va).to_numpy()
    metrics = evaluate(val[TARGET_COL], pred, {"team": val["team_id"], "season": val["season"]})
    return {
        "metrics": metrics,
        "model": model,
        "features": list(reduced),
        "train_medians": {c: float(train_medians[c]) for c in reduced},
    }


# ─────────────────────────────────────────────────────────────
# BIVARIATE (Model1 total + share factor lineal)
# ─────────────────────────────────────────────────────────────

def load_match_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG["match_dataset_path"])
    log.info("Match dataset cargado: %s", df.shape)
    return df


def split_temporal_match(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(CONFIG["train_seasons"])].copy()
    val = df[df["season"].isin(CONFIG["val_seasons"])].copy()
    test = df[df["season"].isin(CONFIG["test_seasons"])].copy()
    assert train["match_id"].isin(val["match_id"]).sum() == 0
    log.info("Match split — train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test


def get_match_feature_columns(df: pd.DataFrame) -> list[str]:
    """Features para Model1: numéricas excepto IDs, target, targets auxiliares y flags no numéricos."""
    exclude = {
        "match_id", "season", "match_date",
        "home_team_id", "away_team_id", "home_team_name", "away_team_name",
        # Targets y diagnóstico (no se usan como features)
        "throw_ins_total_match", "home_throw_ins_total", "away_throw_ins_total",
        "share_home",
    }
    feats = []
    for c in df.columns:
        if c in exclude:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        feats.append(c)
    return feats


def train_lightgbm_total(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> lgb.LGBMRegressor:
    params = dict(CONFIG["lgb_params"])
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(CONFIG["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _compute_share_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las features del modelo de share: possession_diff y home_rolling_diff."""
    out = pd.DataFrame(index=df.index)
    out["possession_diff"] = (
        df["home_rolling5_possession_pct"].fillna(df["home_rolling5_possession_pct"].median())
        - df["away_rolling5_possession_pct"].fillna(df["away_rolling5_possession_pct"].median())
    )
    out["home_rolling_diff"] = (
        df["home_rolling5_throw_ins_total"].fillna(df["home_rolling5_throw_ins_total"].median())
        - df["away_rolling5_throw_ins_total"].fillna(df["away_rolling5_throw_ins_total"].median())
    )
    return out


def fit_share_model(train: pd.DataFrame) -> tuple[LinearRegression, list[str]]:
    """Regresión lineal share_home ~ possession_diff + home_rolling_diff."""
    X = _compute_share_features(train)
    y = train["share_home"].astype(float)
    features = list(X.columns)
    model = LinearRegression()
    model.fit(X, y)
    log.info(
        "Share model — intercept=%.4f | coefs=%s",
        model.intercept_,
        {f: float(c) for f, c in zip(features, model.coef_)},
    )
    return model, features


def _evaluate_bivariate(
    val: pd.DataFrame,
    pred_total: np.ndarray,
    pred_share: np.ndarray,
) -> dict:
    """Calcula métricas bivariate + reconstrucción per-team MAE."""
    total_mae = float(mean_absolute_error(val["throw_ins_total_match"], pred_total))
    total_rmse = float(np.sqrt(mean_squared_error(val["throw_ins_total_match"], pred_total)))

    share_mae = float(mean_absolute_error(val["share_home"], pred_share))

    pred_home = pred_total * pred_share
    pred_away = pred_total * (1.0 - pred_share)

    y_home = val["home_throw_ins_total"].astype(float).to_numpy()
    y_away = val["away_throw_ins_total"].astype(float).to_numpy()

    per_team_mae = float(
        (np.abs(pred_home - y_home).sum() + np.abs(pred_away - y_away).sum())
        / (len(pred_home) + len(pred_away))
    )
    per_team_rmse = float(
        np.sqrt(
            (np.square(pred_home - y_home).sum() + np.square(pred_away - y_away).sum())
            / (len(pred_home) + len(pred_away))
        )
    )

    return {
        "total_mae": total_mae,
        "total_rmse": total_rmse,
        "share_mae": share_mae,
        "share_pred_mean": float(np.mean(pred_share)),
        "share_pred_std": float(np.std(pred_share)),
        "share_actual_mean": float(val["share_home"].mean()),
        "share_actual_std": float(val["share_home"].std()),
        "per_team_mae_reconstructed": per_team_mae,
        "per_team_rmse_reconstructed": per_team_rmse,
    }


def main_bivariate(weights_mode: str = "both") -> None:
    log.info("=== Modo BIVARIATE: Model1(total) + share lineal ===")
    df = load_match_dataset()
    train, val, _test = split_temporal_match(df)

    feature_cols = get_match_feature_columns(df)
    log.info("Features match-level: %d", len(feature_cols))

    X_train = train[feature_cols].astype(float)
    y_train = train["throw_ins_total_match"].astype(float)
    X_val = val[feature_cols].astype(float)
    y_val = val["throw_ins_total_match"].astype(float)

    # Sample weights: se aplica el mismo esquema que en modo per-team
    schemes = ["uniform", "decay"] if weights_mode == "both" else [weights_mode]

    total_results: dict[str, dict] = {}
    total_models: dict[str, lgb.LGBMRegressor] = {}
    for scheme in schemes:
        log.info("Entrenando Model1 (total) weights=%s ...", scheme)
        sw = None
        if scheme == "decay":
            sw = train["season"].map(CONFIG["season_decay_weights"]).fillna(1.0).to_numpy()
        model = train_lightgbm_total(X_train, y_train, X_val, y_val, sw)
        pred_val = model.predict(X_val)
        pred_train = model.predict(X_train)
        total_results[scheme] = {
            "total_mae": float(mean_absolute_error(y_val, pred_val)),
            "total_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
            "train_mae": float(mean_absolute_error(y_train, pred_train)),
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
        }
        total_results[scheme]["train_val_gap"] = round(
            total_results[scheme]["train_mae"] / total_results[scheme]["total_mae"], 4
        )
        log.info(
            "  weights=%s → train_MAE %.4f | total_MAE val %.4f | gap %.4f | best_iter %d",
            scheme, total_results[scheme]["train_mae"],
            total_results[scheme]["total_mae"],
            total_results[scheme]["train_val_gap"],
            total_results[scheme]["best_iteration"],
        )
        total_models[scheme] = model

    best_scheme = min(total_models, key=lambda s: total_results[s]["total_mae"])
    best_total_model = total_models[best_scheme]
    log.info("Mejor esquema Model1: %s (total MAE %.4f)",
             best_scheme, total_results[best_scheme]["total_mae"])

    # Share factor lineal
    log.info("Entrenando share factor (regresión lineal) ...")
    share_model, share_features = fit_share_model(train)

    X_share_val = _compute_share_features(val)
    pred_share_val = share_model.predict(X_share_val)
    # Clip a [0, 1] para evitar predicciones fuera de rango (lineal puede salirse)
    pred_share_val = np.clip(pred_share_val, 0.0, 1.0)

    # Sanity: mean pred_share ≈ mean real (≈ 0.51)
    log.info(
        "share val — pred mean %.4f (std %.4f) vs real mean %.4f (std %.4f)",
        pred_share_val.mean(), pred_share_val.std(),
        val["share_home"].mean(), val["share_home"].std(),
    )

    # Reconstrucción bivariate
    pred_total_val = best_total_model.predict(X_val)
    biv_metrics = _evaluate_bivariate(val, pred_total_val, pred_share_val)
    biv_metrics["best_total_scheme"] = best_scheme
    biv_metrics["total_train_mae"] = total_results[best_scheme]["train_mae"]
    biv_metrics["total_train_val_gap"] = total_results[best_scheme]["train_val_gap"]
    biv_metrics["total_best_iteration"] = total_results[best_scheme]["best_iteration"]
    biv_metrics["current_per_team_mae"] = CONFIG["current_per_team_mae"]
    biv_metrics["beats_current"] = (
        biv_metrics["per_team_mae_reconstructed"] < CONFIG["current_per_team_mae"]
    )

    decision = "MEJORA" if biv_metrics["beats_current"] else "NO MEJORA"
    log.info(
        "=> per_team_MAE reconstruido %.4f vs current %.4f → [%s]",
        biv_metrics["per_team_mae_reconstructed"],
        CONFIG["current_per_team_mae"],
        decision,
    )

    # Persistencia
    Path(CONFIG["model_total_path"]).parent.mkdir(parents=True, exist_ok=True)

    total_artifact = {
        "model": best_total_model,
        "version": "v1_total",
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "features": feature_cols,
        "params": dict(CONFIG["lgb_params"]),
        "val_total_mae": total_results[best_scheme]["total_mae"],
        "val_total_rmse": total_results[best_scheme]["total_rmse"],
        "sample_weights_scheme": best_scheme,
        "train_seasons": CONFIG["train_seasons"],
        "val_seasons": CONFIG["val_seasons"],
        "best_iteration": total_results[best_scheme]["best_iteration"],
    }
    joblib.dump(total_artifact, CONFIG["model_total_path"])
    log.info("Model1 guardado: %s", CONFIG["model_total_path"])

    share_coefs = {
        "intercept": float(share_model.intercept_),
        "coefs": {f: float(c) for f, c in zip(share_features, share_model.coef_)},
        "features": share_features,
        "trained_on": "+".join(CONFIG["train_seasons"]),
        "clip_range": [0.0, 1.0],
    }
    with open(CONFIG["share_coefs_path"], "w", encoding="utf-8") as f:
        json.dump(share_coefs, f, indent=2, ensure_ascii=False)
    log.info("Share coefs guardados: %s", CONFIG["share_coefs_path"])

    def _to_native(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"No serializable: {type(obj).__name__}")

    metrics_out = {
        "trained_at": total_artifact["trained_at"],
        "mode": "bivariate",
        "n_features": len(feature_cols),
        "train_rows": len(train),
        "val_rows": len(val),
        "total_results_by_scheme": total_results,
        "bivariate": biv_metrics,
        "share_coefs": share_coefs,
    }
    with open(CONFIG["metrics_bivariate_path"], "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, ensure_ascii=False, default=_to_native)
    log.info("Métricas bivariate guardadas: %s", CONFIG["metrics_bivariate_path"])

    # Feature importance del Model1
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": best_total_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": best_total_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(CONFIG["feature_importance_total_path"], index=False)
    log.info("Feature importance Model1 guardado: %s", CONFIG["feature_importance_total_path"])
    log.info("Top 10 features Model1 por gain:\n%s", fi.head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# TEAM BIAS CALIBRATION — regeneración post-training
# ─────────────────────────────────────────────────────────────

def _regenerate_team_bias_calibration(
    model: lgb.LGBMRegressor,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    model_trained_at: str,
    model_train_seasons: list[str],
    out_path: str | Path | None = None,
    backup_path: str | Path | None = None,
    shrinkage_k: float | None = None,
    prior_mu: float | None = None,
) -> dict:
    """
    Regenera `team_bias_calibration_v2.json` desde residuos post-training.

    Calcula, por cada (team_id, is_home) presente en `val_df`:
      - n: nº de matches en val
      - raw_bias: media de residuos (y_true - lambda_pred)
      - shrunk_bias: posterior bayesiano usando `model.market_utils.shrink_bias`

    Prior (reverse-engineered del JSON frozen del 2026-04-21):
      prior_mu = 0, k = sigma2_res / sigma2_prior = 5.0.

    Contrato:
      - Usa SOLO `market_utils.shrink_bias` para la matemática de shrinkage
        (invariante ADR D2 — no reimplementar inline).
      - Escritura atómica: json.dump a `.tmp`, luego os.replace.
      - Backup: si `backup_path` no existe todavía, copia el archivo actual allí.
        Si ya existe, NO se sobrescribe (el backup es baseline histórico).
      - Schema idéntico al JSON existente: corrections[tid][hid] = {n, raw_bias, shrunk_bias}.
      - Top-level: description, generated_at, model_trained_at, model_train_seasons.

    Args:
        model: LGBMRegressor entrenado (debe tener predict()).
        val_df: validation DataFrame con columnas team_id, is_home, TARGET_COL + feature_cols.
        feature_cols: lista de features exactamente como las usa el modelo.
        model_trained_at: timestamp del modelo (joblib artifact["trained_at"]).
        model_train_seasons: temporadas de entrenamiento (para metadata).
        out_path: destino del JSON. Default CONFIG["team_bias_path"].
        backup_path: pre-regen backup. Default CONFIG["team_bias_pre_regen_bak_path"].
        shrinkage_k: override del ratio k. Default CONFIG["team_bias_shrinkage_k"]=5.0.
        prior_mu: override del prior mu. Default CONFIG["team_bias_prior_mu"]=0.0.

    Returns:
        El payload escrito (dict — útil para test/smoke).
    """
    import os
    import shutil

    out_p = Path(out_path or CONFIG["team_bias_path"])
    bak_p = Path(backup_path or CONFIG["team_bias_pre_regen_bak_path"])
    k = float(shrinkage_k if shrinkage_k is not None else CONFIG["team_bias_shrinkage_k"])
    mu = float(prior_mu if prior_mu is not None else CONFIG["team_bias_prior_mu"])

    # 1. Backup defensivo (solo si no existe). El backup es baseline inmutable.
    if out_p.exists() and not bak_p.exists():
        bak_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_p, bak_p)
        log.info("team_bias backup creado: %s", bak_p)

    # 2. Predecir sobre val y computar residuos.
    X_val = val_df[feature_cols].astype(float)
    y_true = val_df[TARGET_COL].astype(float).to_numpy()
    lam_pred = np.asarray(model.predict(X_val), dtype=float)
    residuals = y_true - lam_pred

    # 3. Agregar por (team_id, is_home) y aplicar shrinkage.
    agg_df = pd.DataFrame({
        "team_id": val_df["team_id"].to_numpy(),
        "is_home": val_df["is_home"].to_numpy(),
        "residual": residuals,
    })
    grouped = agg_df.groupby(["team_id", "is_home"])["residual"].agg(["count", "mean"])

    corrections: dict[str, dict[str, dict[str, float]]] = {}
    raw_abs: list[float] = []
    shr_abs: list[float] = []
    for (tid, hid), row in grouped.iterrows():
        n = int(row["count"])
        raw = float(row["mean"])
        shrunk = shrink_bias(raw, n, k=k, prior_mu=mu)
        corrections.setdefault(str(int(tid)), {})[str(int(hid))] = {
            "n": n,
            "raw_bias": round(raw, 4),
            "shrunk_bias": round(shrunk, 4),
        }
        raw_abs.append(abs(raw))
        shr_abs.append(abs(shrunk))

    payload = {
        "description": (
            "Sesgo post-hoc por (team_id, is_home) con shrinkage bayesiano — "
            "regenerado desde model/train.py (prior normal-normal, k=%.2f, mu=%.2f)" % (k, mu)
        ),
        "generated_at": datetime.utcnow().isoformat() + "+00:00",
        "model_trained_at": model_trained_at,
        "model_train_seasons": list(model_train_seasons),
        "shrinkage_k": k,
        "prior_mu": mu,
        "corrections": corrections,
    }

    # 4. Escritura atómica (tmp + os.replace). json.dump con sort_keys por idempotencia.
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp_p, out_p)

    mean_raw = float(np.mean(raw_abs)) if raw_abs else 0.0
    mean_shr = float(np.mean(shr_abs)) if shr_abs else 0.0
    log.info(
        "team_bias_regen teams=%d mean_abs_raw_bias=%.4f mean_abs_shrunk_bias=%.4f "
        "prior_mu=%.4f shrinkage_k=%.4f out=%s",
        len(corrections), mean_raw, mean_shr, mu, k, out_p,
    )
    return payload


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(weights_mode: str = "both", features_mode: str = "shap30") -> None:
    df = load_dataset()
    train, val, test = split_temporal(df)

    baseline_mae = baseline_team_mean(train, val)

    if features_mode == "shap30":
        missing = [f for f in SHAP_SELECTED_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"SHAP_SELECTED_FEATURES contiene columnas ausentes: {missing}")
        feature_cols = SHAP_SELECTED_FEATURES
        log.info("Modo shap30: usando %d features seleccionadas por SHAP", len(feature_cols))
    else:
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
        log.info("Modo all: features tras filtrar raw match stats: %d", len(feature_cols))

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

        train_pred = model.predict(X_train)
        metrics["train_mae"] = float(mean_absolute_error(y_train, train_pred))
        metrics["train_val_gap"] = round(metrics["train_mae"] / metrics["mae"], 4)

        log.info(
            "  weights=%s → train_MAE %.4f | val_MAE %.4f | gap %.4f | best_iter %d",
            scheme, metrics["train_mae"], metrics["mae"],
            metrics["train_val_gap"], metrics["best_iteration"],
        )
        results[f"lgbm_{scheme}"] = metrics
        models[scheme] = model

    log.info("Entrenando NegBinom sanity check ...")
    negbinom_out = train_negbinom(train, val)
    negbinom_model = None
    negbinom_features: list[str] = []
    negbinom_train_medians: dict[str, float] = {}
    if negbinom_out:
        # Nuevo contrato: dict con keys metrics/model/features/train_medians.
        nb_metrics = negbinom_out["metrics"]
        negbinom_model = negbinom_out["model"]
        negbinom_features = list(negbinom_out["features"])
        negbinom_train_medians = dict(negbinom_out["train_medians"])
        results["negbinom"] = nb_metrics
        log.info("  negbinom → MAE %.4f | RMSE %.4f", nb_metrics["mae"], nb_metrics["rmse"])

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

    # Ensemble-ready artifact: se preserva `model` (backward compat con predict.py)
    # y además se persisten los 3 modelos bajo `models` (LGBM uniform + LGBM decay + NegBinom GLM)
    # para el ensemble ponderado que computa el total en inferencia.
    models_bundle: dict[str, object] = {}
    for scheme_name, mdl in models.items():
        models_bundle[f"lgbm_{scheme_name}"] = mdl
    if negbinom_model is not None:
        models_bundle["negbinom"] = negbinom_model

    artifact = {
        "model": best_model,
        "models": models_bundle,
        "version": "v1",
        "trained_at": datetime.utcnow().isoformat(timespec="seconds"),
        "features": feature_cols,
        "negbinom_features": negbinom_features,
        "negbinom_train_medians": negbinom_train_medians,
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

    # Regeneración de team_bias_calibration_v2.json usando los residuos
    # del modelo recién entrenado sobre el mismo `val`. Único code path que
    # escribe este JSON (ADR D2 — helper compartido en market_utils.shrink_bias).
    try:
        _regenerate_team_bias_calibration(
            model=best_model,
            val_df=val,
            feature_cols=feature_cols,
            model_trained_at=artifact["trained_at"],
            model_train_seasons=CONFIG["train_seasons"],
        )
    except Exception as exc:
        log.error("team_bias_regen falló (JSON anterior intacto): %s", exc)

    metrics_out = {
        "trained_at": artifact["trained_at"],
        "baseline_mae": baseline_mae,
        "baseline_target_mae": CONFIG["baseline_target_mae"],
        "best_model": f"lgbm_{best_scheme}",
        "beats_baseline": success,
        "n_features": len(feature_cols),
        "n_features_used": len(feature_cols),
        "feature_selection_method": features_mode,
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


# ─────────────────────────────────────────────────────────────
# WALK-FORWARD CROSS-VALIDATION (diagnóstico)
# ─────────────────────────────────────────────────────────────

def _build_decay_weights(train_seasons: list[str]) -> dict[str, float]:
    """
    Devuelve pesos por temporada: la más reciente = 1.0, cada anterior decae 0.2.
    Mínimo clip 0.2 para que las temporadas más antiguas sigan contando.
    Uso agnóstico del fold (en vez del `season_decay_weights` hardcodeado).
    """
    seasons_sorted = sorted(train_seasons)
    weights = {}
    n = len(seasons_sorted)
    for i, s in enumerate(seasons_sorted):
        w = 1.0 - 0.2 * (n - 1 - i)
        weights[s] = max(w, 0.2)
    return weights


def _decay_sample_weights(train: pd.DataFrame, weights_map: dict[str, float]) -> np.ndarray:
    return train["season"].map(weights_map).fillna(1.0).to_numpy()


def run_walk_forward_cv(features_mode: str = "shap30") -> dict:
    """
    Ejecuta walk-forward CV de 3 folds (expanding window) como diagnóstico.
    No sobrescribe `model_v1.joblib` ni `metrics_v1.json`.
    """
    df = load_dataset()

    if features_mode == "shap30":
        missing = [f for f in SHAP_SELECTED_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"SHAP_SELECTED_FEATURES contiene columnas ausentes: {missing}")
        feature_cols = SHAP_SELECTED_FEATURES
    else:
        feature_cols = get_feature_columns(df)
    log.info("CV — features_mode=%s, n_features=%d", features_mode, len(feature_cols))

    folds_out: list[dict] = []
    for fold_cfg in CONFIG["cv_folds"]:
        name = fold_cfg["name"]
        train_seasons = fold_cfg["train_seasons"]
        val_season = fold_cfg["val_season"]
        is_partial = fold_cfg["is_partial"]

        train = df[df["season"].isin(train_seasons)].copy()
        val = df[df["season"] == val_season].copy()

        if train.empty or val.empty:
            log.warning("Fold %s saltado (train=%d, val=%d)", name, len(train), len(val))
            continue

        log.info(
            "=== Fold %s — train=%s (%d filas) | val=%s (%d filas)%s ===",
            name, "+".join(train_seasons), len(train), val_season, len(val),
            " [PARTIAL]" if is_partial else "",
        )

        baseline_mae = baseline_team_mean(train, val)

        X_train = train[feature_cols]
        y_train = train[TARGET_COL]
        X_val = val[feature_cols]
        y_val = val[TARGET_COL]

        decay_map = _build_decay_weights(train_seasons)
        sw = _decay_sample_weights(train, decay_map)

        model = train_lightgbm(X_train, y_train, X_val, y_val, sw)
        val_pred = model.predict(X_val)
        train_pred = model.predict(X_train)

        val_mae = float(mean_absolute_error(y_val, val_pred))
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))
        train_mae = float(mean_absolute_error(y_train, train_pred))
        gap = round(train_mae / val_mae, 4) if val_mae > 0 else 0.0
        best_iter = int(model.best_iteration_ or model.n_estimators)

        log.info(
            "  %s → val_MAE %.4f | val_RMSE %.4f | train_MAE %.4f | gap %.4f | "
            "best_iter %d | baseline %.4f",
            name, val_mae, val_rmse, train_mae, gap, best_iter, baseline_mae,
        )

        folds_out.append({
            "name": name,
            "train_seasons": train_seasons,
            "val_season": val_season,
            "is_partial": is_partial,
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "val_mae": round(val_mae, 4),
            "val_rmse": round(val_rmse, 4),
            "train_mae": round(train_mae, 4),
            "train_val_gap": gap,
            "best_iteration": best_iter,
            "baseline_mae": round(float(baseline_mae), 4),
            "decay_weights": {k: round(v, 2) for k, v in decay_map.items()},
        })

    # Agregados solo sobre folds completos
    complete = [f for f in folds_out if not f["is_partial"]]
    maes = [f["val_mae"] for f in complete]
    if maes:
        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes, ddof=0))
        min_mae = float(np.min(maes))
        max_mae = float(np.max(maes))
        rng = round(max_mae - min_mae, 4)
    else:
        mean_mae = std_mae = min_mae = max_mae = rng = 0.0

    aggregated = {
        "n_complete_folds": len(complete),
        "mean_val_mae": round(mean_mae, 4),
        "std_val_mae": round(std_mae, 4),
        "min_val_mae": round(min_mae, 4),
        "max_val_mae": round(max_mae, 4),
        "range_val_mae": rng,
    }

    significant_delta = round(2 * std_mae, 4)
    log.info(
        "CV AGREGADO — mean %.4f | std %.4f | range %.4f (sobre %d folds completos)",
        mean_mae, std_mae, rng, len(complete),
    )
    log.info(
        "→ std_val_MAE = %.4f → cambios con delta < %.4f son ruido estadístico "
        "(no significativos a 2σ)", std_mae, significant_delta,
    )

    out = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "features_mode": features_mode,
        "n_features": len(feature_cols),
        "lgb_params": dict(CONFIG["lgb_params"]),
        "early_stopping_rounds": CONFIG["early_stopping_rounds"],
        "folds": folds_out,
        "aggregated": aggregated,
        "significant_delta_2sigma": significant_delta,
    }

    def _to_native(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"No serializable: {type(obj).__name__}")

    out_dir = Path(CONFIG["cv_results_path"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(CONFIG["cv_results_path"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=_to_native)
    log.info("CV results guardados: %s", CONFIG["cv_results_path"])

    # CSV legible: 1 fila por fold + fila final con agregados
    rows = []
    for f_ in folds_out:
        rows.append({
            "fold": f_["name"],
            "train_seasons": "+".join(f_["train_seasons"]),
            "val_season": f_["val_season"],
            "is_partial": f_["is_partial"],
            "n_train": f_["n_train"],
            "n_val": f_["n_val"],
            "val_mae": f_["val_mae"],
            "val_rmse": f_["val_rmse"],
            "train_mae": f_["train_mae"],
            "train_val_gap": f_["train_val_gap"],
            "best_iteration": f_["best_iteration"],
            "baseline_mae": f_["baseline_mae"],
        })
    rows.append({
        "fold": "AGGREGATED (complete only)",
        "train_seasons": "",
        "val_season": "",
        "is_partial": "",
        "n_train": "",
        "n_val": "",
        "val_mae": aggregated["mean_val_mae"],
        "val_rmse": "",
        "train_mae": "",
        "train_val_gap": "",
        "best_iteration": "",
        "baseline_mae": f"std={aggregated['std_val_mae']} range={aggregated['range_val_mae']}",
    })
    pd.DataFrame(rows).to_csv(CONFIG["cv_results_csv_path"], index=False)
    log.info("CV CSV guardado: %s", CONFIG["cv_results_csv_path"])

    return out


def run_experiment_cv(
    feature_list: list[str],
    label: str = "experiment",
    features_mode: str = "custom",
    lgb_params_override: dict | None = None,
) -> dict:
    """
    Evalúa una lista de features sobre los 5 folds de walk-forward CV.

    Devuelve mean_val_MAE ± std (para compatibilidad) y además el DataFrame de
    predicciones pooled de todos los folds completos, necesario para
    compare_experiments_bootstrap().

    Uso recomendado:

        baseline  = run_experiment_cv(SHAP_SELECTED_FEATURES, label="baseline")
        candidate = run_experiment_cv(SHAP_SELECTED_FEATURES + ["new_feat"], label="new_feat")
        cmp = compare_experiments_bootstrap(baseline, candidate)
        print(f"Δ={cmp['delta_mean']:+.4f}  IC95%=[{cmp['delta_ci_low']:+.4f}, {cmp['delta_ci_high']:+.4f}]  sig={cmp['significant']}")

    Args:
        feature_list:  lista de columnas a usar como features.
        label:         nombre del experimento (aparece en logs y en el JSON de salida).
        features_mode: string descriptivo del método de selección (solo para metadata).

    Returns:
        dict con mean_val_mae, std_val_mae, significant_delta_2sigma, folds_detail,
        y además result["predictions"] (DataFrame con match_id, is_home, y_true, y_pred, fold).
    """
    df = load_dataset()

    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        raise ValueError(f"run_experiment_cv: features ausentes en dataset: {missing}")

    log.info("=== Experimento CV: %s | %d features ===", label, len(feature_list))

    folds_out: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for fold_cfg in CONFIG["cv_folds"]:
        name = fold_cfg["name"]
        train_seasons = fold_cfg["train_seasons"]
        val_season = fold_cfg["val_season"]
        is_partial = fold_cfg["is_partial"]

        train = df[df["season"].isin(train_seasons)].copy()
        val = df[df["season"] == val_season].copy()

        if train.empty or val.empty:
            log.warning("Fold %s saltado (train=%d, val=%d)", name, len(train), len(val))
            continue

        decay_map = _build_decay_weights(train_seasons)
        sw = _decay_sample_weights(train, decay_map)

        model = train_lightgbm(
            train[feature_list], train[TARGET_COL],
            val[feature_list], val[TARGET_COL],
            sw,
            lgb_params_override=lgb_params_override,
        )
        preds = model.predict(val[feature_list])
        val_mae = float(mean_absolute_error(val[TARGET_COL], preds))
        log.info("  %s → val_MAE %.4f%s", name, val_mae, " [PARTIAL]" if is_partial else "")

        folds_out.append({
            "name": name,
            "val_season": val_season,
            "is_partial": is_partial,
            "val_mae": round(val_mae, 4),
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
    significant_delta = round(2 * std_mae, 4)

    predictions_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    log.info(
        "=== %s — mean_MAE=%.4f | std=%.4f | 2σ=%.4f | n_preds=%d ===",
        label, mean_mae, std_mae, significant_delta, len(predictions_df),
    )

    return {
        "label": label,
        "n_features": len(feature_list),
        "features_mode": features_mode,
        "n_complete_folds": len(complete),
        "mean_val_mae": round(mean_mae, 4),
        "std_val_mae": round(std_mae, 4),
        "significant_delta_2sigma": significant_delta,
        "folds": folds_out,
        "predictions": predictions_df,
    }


def compare_experiments_bootstrap(
    result_a: dict,
    result_b: dict,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Paired cluster bootstrap para comparar dos experimentos de CV.

    Resamplea partidos completos (cluster = match_id, que agrupa home + away) con
    reemplazo y calcula Δ MAE = MAE(b) - MAE(a) en cada iteración. Un valor
    negativo significa que B es mejor (menor MAE).

    Args:
        result_a:  dict devuelto por run_experiment_cv() (baseline).
        result_b:  dict devuelto por run_experiment_cv() (candidato).
        n_boot:    número de iteraciones bootstrap (default 2000).
        seed:      semilla para reproducibilidad.

    Returns:
        dict con:
          delta_mean      — Δ MAE medio observado (b - a); negativo = b mejor
          delta_ci_low    — percentil 2.5% del bootstrap
          delta_ci_high   — percentil 97.5% del bootstrap
          p_value         — fracción de bootstrap donde Δ > 0 (o < 0 si delta_mean < 0)
          significant     — True si el IC 95% no cruza 0
          ci_halfwidth    — (delta_ci_high - delta_ci_low) / 2; umbral efectivo
          n_matches       — número de partidos únicos en el pool
          n_rows          — número de filas (home+away) en el pool
    """
    preds_a = result_a.get("predictions")
    preds_b = result_b.get("predictions")

    if preds_a is None or preds_b is None or preds_a.empty or preds_b.empty:
        raise ValueError(
            "compare_experiments_bootstrap: ambos resultados deben contener 'predictions'. "
            "Usa run_experiment_cv() para generarlos."
        )

    merged = preds_a[["match_id", "is_home", "y_true", "y_pred"]].merge(
        preds_b[["match_id", "is_home", "y_pred"]].rename(columns={"y_pred": "y_pred_b"}),
        on=["match_id", "is_home"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("compare_experiments_bootstrap: sin partidos comunes entre experimentos.")

    match_ids = merged["match_id"].unique()
    n_matches = len(match_ids)
    rng = np.random.default_rng(seed)

    delta_obs = float(
        mean_absolute_error(merged["y_true"], merged["y_pred_b"])
        - mean_absolute_error(merged["y_true"], merged["y_pred"])
    )

    boot_deltas = np.empty(n_boot)
    for i in range(n_boot):
        sampled_ids = rng.choice(match_ids, size=n_matches, replace=True)
        mask = merged["match_id"].isin(sampled_ids)
        sub = merged[mask]
        mae_a = np.mean(np.abs(sub["y_true"].values - sub["y_pred"].values))
        mae_b = np.mean(np.abs(sub["y_true"].values - sub["y_pred_b"].values))
        boot_deltas[i] = mae_b - mae_a

    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))
    significant = not (ci_low <= 0 <= ci_high)

    if delta_obs < 0:
        p_value = float(np.mean(boot_deltas >= 0))
    else:
        p_value = float(np.mean(boot_deltas <= 0))

    ci_halfwidth = round((ci_high - ci_low) / 2, 4)

    log.info(
        "Bootstrap [%s vs %s] delta=%+.4f  IC95=[%+.4f, %+.4f]  p=%.3f  sig=%s  halfwidth=%.4f",
        result_a["label"], result_b["label"],
        delta_obs, ci_low, ci_high, p_value, significant, ci_halfwidth,
    )

    return {
        "label_a": result_a["label"],
        "label_b": result_b["label"],
        "delta_mean": round(delta_obs, 4),
        "delta_ci_low": round(ci_low, 4),
        "delta_ci_high": round(ci_high, 4),
        "p_value": round(p_value, 4),
        "significant": significant,
        "ci_halfwidth": ci_halfwidth,
        "n_matches": n_matches,
        "n_rows": len(merged),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train throw-in predictor")
    parser.add_argument(
        "--weights",
        choices=["uniform", "decay", "both"],
        default="both",
        help="Esquema de sample weights a probar",
    )
    parser.add_argument(
        "--target",
        choices=["per-team", "bivariate"],
        default="per-team",
        help="Target del modelo: per-team (current) o bivariate (Model1 total + share)",
    )
    parser.add_argument(
        "--features",
        choices=["all", "shap30"],
        default="shap30",
        help="Selección de features: shap30 (top-30 SHAP, default) o all (101 features)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "cv"],
        default="train",
        help="train: entrena y guarda model_v1.joblib; cv: walk-forward diagnóstico (no toca artefactos de producción)",
    )
    args = parser.parse_args()
    if args.mode == "cv":
        run_walk_forward_cv(features_mode=args.features)
    elif args.target == "bivariate":
        main_bivariate(weights_mode=args.weights)
    else:
        main(weights_mode=args.weights, features_mode=args.features)
