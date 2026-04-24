"""
Tune α (NegBin dispersion) via MLE + sanity MoM — investigación offline
=========================================================================
Investigación pura (NO toca producción). Ajusta el hiperparámetro α del modelo
NegBin(μ, α) para probabilidades de tail O/U. La media μ se calcula con la
MISMA lógica que usa producción hoy (ver invariante `architecture/
model-inference-conventions`):

    μ_match = apply_team_bias(lgbm_uniform.predict(X_home)) +
              apply_team_bias(lgbm_uniform.predict(X_away))

(equivalente a `pred_total_v2 = pred_home_v2 + pred_away_v2` en `predict.py`,
rama `_uniform_only` con weights {1.0, 0.0, 0.0}).

## Metodología — PRIMARY: Grid-search MLE

Para cada α en un grid [0.01, 0.02, ..., 2.00] evaluamos la log-likelihood
NegBin del conjunto de totales observados en val (2025/2026):

    LL(α) = Σ_i log P(y_i | μ_i, α)

con `scipy.stats.nbinom.logpmf(y, n=1/α, p=1/(1+α·μ))` (parametrización
verificada en `test_mean_matched_variance.py`). Maximiza sobre el grid.

## Sanity — SECONDARY: Method-of-moments

Usando la relación:

    Var(y | μ) = μ + α·μ²     ⇒    α_MoM ≈ Σ((y-μ)² - μ) / Σ(μ²)

Reportamos ambas estimaciones y un `dispersion_ratio = Var(y|μ)/μ`. Si
α_MLE está fuera de [0.01, 2.0], flageamos anomalía y el script devuelve
exit=2 (para bloquear Phase C).

## Invariantes / constraints (HARD)

- NO reentrena nada. Lee `data/model/model_v1.joblib` read-only.
- NO modifica el joblib. NO escribe en `data/model/predictions/`.
- Output: `data/model/_investigation_tuned_alpha.json` + stdout.
- Usa `model.market_utils.{load_team_bias, apply_team_bias}` como único
  camino para calibrar λ per-team (single-source invariant).

## Uso

  python -m scripts.investigation.tune_alpha_via_cv

## Output JSON (contrato con Phase C)

  {
    "method": "grid_mle",
    "alpha_tuned": 0.XX,
    "alpha_grid": [...],
    "log_likelihood_per_alpha": {...},
    "val_n": XXX,
    "val_mae": X.XX,
    "var_observed": X.XX,
    "var_poisson_implied": X.XX,
    "dispersion_ratio": X.XX,
    "alpha_mom": X.XX,
    "trained_at": "ISO-8601"
  }
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.market_utils import apply_team_bias, load_team_bias  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("tune_alpha")

MODEL_PATH = _root / "data/model/model_v1.joblib"
DATASET_PATH = _root / "data/model/dataset.parquet"
BIAS_PATH = _root / "data/model/team_bias_calibration_v2.json"
OUTPUT_JSON = _root / "data/model/_investigation_tuned_alpha.json"

VAL_SEASON = "2025/2026"
TARGET_COL = "throw_ins_total"

# Grid con resolución fina cerca de 0 (donde cae la varianza empírica de
# saques de banda bajo la producción actual) y más gruesa arriba: 0.001..0.05
# step 0.001, luego 0.05..2.00 step 0.01. Total ~245 puntos. La razón de la
# resolución fina: con dispersion_ratio ~1.3 la LL es muy sensible en [0, 0.05]
# y casi plana de 0.1 en adelante — sin esta densidad bajamos el óptimo al
# borde artificial.
_ALPHA_GRID_FINE = np.round(np.arange(0.001, 0.050, 0.001), 4)
_ALPHA_GRID_COARSE = np.round(np.arange(0.050, 2.01, 0.01), 4)
ALPHA_GRID = sorted(set(_ALPHA_GRID_FINE.tolist() + _ALPHA_GRID_COARSE.tolist()))
ALPHA_SANITY_RANGE = (0.001, 2.0)


def _negbin_logpmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """log P(Y=y | NegBin(μ, α)) vectorizado.

    Parametrización scipy: n = 1/α, p = 1/(1 + α·μ). Verificado en
    test_mean_matched_variance._negbin_params: mean=μ, var=μ+α·μ².
    """
    if alpha <= 0:
        raise ValueError(f"alpha debe ser > 0, recibido {alpha}")
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return nbinom.logpmf(y, n, p)


def _compute_val_mu_per_match(art: dict) -> pd.DataFrame:
    """Reproduce producción: μ_match = bias(lam_home) + bias(lam_away).

    Devuelve DataFrame con columnas [match_id, mu, y_total]. y_total es la
    suma de throw_ins_total por match_id (home + away observado).
    """
    model = art["models"]["lgbm_uniform"]
    feats = art["features"]

    df = pd.read_parquet(DATASET_PATH)
    val = df[df["season"] == VAL_SEASON].copy()
    if val.empty:
        raise RuntimeError(f"val season {VAL_SEASON} vacía en {DATASET_PATH}")

    # Predicción per-team (sin bias).
    X_val = val[feats].astype(float)
    lam_raw = np.asarray(model.predict(X_val), dtype=float)

    # Bias per-team (fuente única: market_utils).
    bias_table = load_team_bias(path=BIAS_PATH, model_trained_at=art.get("trained_at"))
    lam_cal = apply_team_bias(
        lam=lam_raw,
        team_id=val["team_id"].to_numpy(),
        is_home=val["is_home"].to_numpy(),
        bias_table=bias_table,
    )
    lam_cal = np.asarray(lam_cal, dtype=float)

    # Agregamos por match_id: μ_match = Σ λ_team (home + away).
    per_team = pd.DataFrame({
        "match_id": val["match_id"].to_numpy(),
        "lam": lam_cal,
        "y": val[TARGET_COL].astype(float).to_numpy(),
    })
    agg = per_team.groupby("match_id", as_index=False).agg(mu=("lam", "sum"), y_total=("y", "sum"))
    return agg


def _mom_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """α Method-of-Moments: resuelve Var(y|μ) = μ + α·μ² en el agregado.

        α_MoM = Σ((y-μ)² - μ) / Σ(μ²)

    Si el numerador es ≤ 0 (underdispersion pura), devuelve el límite
    inferior del grid sanity range (no-op: no hay dispersión extra).
    """
    resid_sq = (y - mu) ** 2
    num = float(np.sum(resid_sq - mu))
    den = float(np.sum(mu**2))
    if den <= 0:
        return float("nan")
    if num <= 0:
        return 0.0  # empíricamente equidispersion o underdispersion
    return num / den


def main() -> int:
    if not MODEL_PATH.exists():
        log.error("Modelo no encontrado: %s", MODEL_PATH)
        return 1
    if not DATASET_PATH.exists():
        log.error("Dataset no encontrado: %s", DATASET_PATH)
        return 1

    art = joblib.load(MODEL_PATH)
    log.info("Modelo: %s trained_at=%s features=%d", art.get("version"), art.get("trained_at"),
             len(art.get("features", [])))

    per_match = _compute_val_mu_per_match(art)
    mu = per_match["mu"].to_numpy(dtype=float)
    y = per_match["y_total"].to_numpy(dtype=float)
    n_val = len(per_match)
    log.info("val matches: %d | μ mean=%.2f std=%.2f | y mean=%.2f std=%.2f",
             n_val, mu.mean(), mu.std(), y.mean(), y.std())

    val_mae = float(np.mean(np.abs(y - mu)))
    var_observed = float(np.var(y - mu, ddof=0))
    var_poisson_implied = float(np.mean(mu))  # E[Var(y|μ)] under Poisson ≈ E[μ]
    dispersion_ratio = var_observed / var_poisson_implied if var_poisson_implied > 0 else float("nan")

    log.info("val_mae=%.4f var_observed=%.2f var_poisson_implied=%.2f dispersion_ratio=%.3f",
             val_mae, var_observed, var_poisson_implied, dispersion_ratio)

    # PRIMARY — MLE grid.
    y_int = np.round(y).astype(int)  # nbinom.logpmf exige enteros
    if not np.all(np.abs(y - y_int) < 1e-9):
        log.warning("y_total no-entero en val (abs max=%.6f) — redondeando para logpmf",
                    float(np.max(np.abs(y - y_int))))

    ll_per_alpha: dict[str, float] = {}
    for a in ALPHA_GRID:
        lp = _negbin_logpmf(y_int, mu, float(a))
        ll = float(np.sum(lp))
        ll_per_alpha[f"{a:.4f}"] = ll

    # Encontrar α* (MLE).
    alpha_best_str = max(ll_per_alpha, key=ll_per_alpha.get)
    alpha_best = float(alpha_best_str)
    ll_best = ll_per_alpha[alpha_best_str]
    log.info("MLE grid: α*=%.4f (LL=%.4f); n_grid=%d", alpha_best, ll_best, len(ALPHA_GRID))

    # Contexto — 5 alphas vecinos (lookup tolerante a jitter floating-point).
    rounded_grid = [round(a, 4) for a in ALPHA_GRID]
    try:
        idx_best = rounded_grid.index(round(alpha_best, 4))
    except ValueError:
        # Fallback: nearest neighbor
        idx_best = int(np.argmin(np.abs(np.asarray(rounded_grid) - alpha_best)))
    window = ALPHA_GRID[max(0, idx_best - 2): min(len(ALPHA_GRID), idx_best + 3)]
    log.info("LL vecindario: %s", {f"{a:.4f}": ll_per_alpha[f"{a:.4f}"] for a in window})

    # SANITY — Method of Moments.
    alpha_mom = _mom_alpha(y, mu)
    log.info("α_MoM (sanity) = %.4f", alpha_mom)

    # Anomalía?
    in_range = ALPHA_SANITY_RANGE[0] <= alpha_best <= ALPHA_SANITY_RANGE[1]
    if not in_range:
        log.error("α*=%.4f FUERA de rango sanity [%.2f, %.2f] — flag anomalía",
                  alpha_best, *ALPHA_SANITY_RANGE)

    # Brier / empirical-var comparison (informativo; no afecta selección).
    var_negbin_implied = float(np.mean(mu + alpha_best * mu**2))
    log.info("Var(y|μ) teórica bajo α*: %.2f (observada: %.2f)", var_negbin_implied, var_observed)

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    payload = {
        "method": "grid_mle",
        "alpha_tuned": round(alpha_best, 4),
        "alpha_grid": ALPHA_GRID,
        "log_likelihood_per_alpha": {k: round(v, 6) for k, v in ll_per_alpha.items()},
        "val_n": int(n_val),
        "val_mae": round(val_mae, 4),
        "var_observed": round(var_observed, 4),
        "var_poisson_implied": round(var_poisson_implied, 4),
        "dispersion_ratio": round(dispersion_ratio, 4),
        "var_negbin_implied_at_best_alpha": round(var_negbin_implied, 4),
        "alpha_mom": round(alpha_mom, 4),
        "alpha_sanity_range": list(ALPHA_SANITY_RANGE),
        "alpha_in_range": bool(in_range),
        "val_season": VAL_SEASON,
        "model_trained_at": art.get("trained_at"),
        "trained_at": trained_at,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    log.info("Guardado: %s", OUTPUT_JSON)

    print()
    print("=" * 80)
    print("TUNED α (MLE grid)")
    print("=" * 80)
    print(f"  α_MLE      = {alpha_best:.4f}   (LL={ll_best:.2f}, n={n_val})")
    print(f"  α_MoM      = {alpha_mom:.4f}   (sanity)")
    print(f"  var_obs    = {var_observed:.2f}")
    print(f"  var_pois   = {var_poisson_implied:.2f}   ratio={dispersion_ratio:.3f}")
    print(f"  var_NB(α*) = {var_negbin_implied:.2f}")
    print(f"  val_MAE    = {val_mae:.4f}")
    print(f"  in_range?  = {in_range}  sanity=[{ALPHA_SANITY_RANGE[0]}, {ALPHA_SANITY_RANGE[1]}]")
    print("=" * 80)

    return 0 if in_range else 2


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
