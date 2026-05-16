"""
Mean-Matched Variance Test — ¿es la underdispersion de Poisson la causa del edge UNDER?
=========================================================================================
Investigación pura (NO toca producción). Aísla el efecto VARIANZA del efecto MEDIA,
completando la refutación del ensemble 0.4/0.4/0.2 (observation #45/#46 en engram).

## Contexto

El ensemble previo (`ensemble-predictions-weighted`) cambió simultáneamente media y
varianza porque NegBin tenía μ=38.3 vs uniform μ=42.5 sobre Sociedad-Getafe. El mean
shift DOMINÓ e incluso AMPLIFICÓ los edges UNDER. Hipótesis nunca testeada aislada.

## Metodología

Para cada (partido, línea O/U) de 2026-04-22:

  1. `p_model_poisson_over = 1 - poisson.cdf(floor(line), μ=pred_total_uniform)`
     — el cálculo de producción actual (single Poisson sobre la media uniform).
  2. `p_model_negbin_over  = 1 - nbinom.cdf(floor(line), n, p)` con MISMA MEDIA
     μ=pred_total_uniform, usando la α del NegBinom entrenado (GLM statsmodels).
     Conversión (μ, α) → (n, p) de scipy:
       n = 1/α
       p = 1/(1 + α·μ)
     Verificación: nbinom.mean(n,p)=μ y nbinom.var(n,p)=μ + α·μ².

Esto congela la MEDIA al valor uniform y varía SOLO la varianza (Poisson var=μ vs
NegBin var=μ+α·μ²). Así podemos medir el impacto puro de la sobredispersión
sobre P(UNDER) y, por ende, sobre el edge.

## Decision tree del brief

- Sociedad-Getafe mean Δedge < -10pp → VARIANZA CONFIRMADA.
- |mean Δedge| < 3pp → GAP ESTRUCTURAL (no varianza).
- entre -3pp y -10pp → PARCIAL.
- mean Δedge > 0 → INESPERADO (posible bug de parametrización).

## Uso

  python -m scripts.investigation.test_mean_matched_variance

## Output

  data/model/_investigation_mean_matched_variance.csv
  Columnas: match, side, line, odds, pred_mean, p_poisson, p_negbin,
            edge_poisson, edge_negbin, delta_edge
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.market_utils import devig_proportional  # type: ignore  # noqa: E402
from scripts.evaluation._normalize import _normalize_team_name  # type: ignore  # noqa: E402

# Aliases manuales — 22bet usa nombres largos ("FC Barcelona", "Atlético de Madrid",
# "Celta de Vigo") que el normalizer estándar no colapsa al nombre corto usado por
# las predicciones. Solo aplicamos estos aliases DENTRO de la investigación; no
# tocamos el pipeline de producción (ver brief HARD constraints).
_INVESTIGATION_ALIASES: dict[str, str] = {
    "fc barcelona": "barcelona",
    "atletico de madrid": "atletico",
    "atlético de madrid": "atletico",
    "celta de vigo": "celta vigo",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("mean_matched_variance")

PREDICTIONS_PATH = _root / "data/model/predictions/predictions_20260422.parquet"
ODDS_PATH = _root / "data/reference/odds_22bet.parquet"
MODEL_PATH = _root / "data/model/model_v1.joblib"
OUTPUT_CSV = _root / "data/model/_investigation_mean_matched_variance.csv"
TARGET_DATE_PREFIX = "2026-04-22"


def _extract_alpha(model_path: Path) -> float:
    """Extrae el α del NegBinom entrenado.

    Busca en orden:
      1. `models["negbinom"].family.alpha` (atributo directo).
      2. Defecto documentado (statsmodels llamado con `NegativeBinomial()` sin arg
         → alpha=1.0).
    """
    art = joblib.load(model_path)
    nb_model = art.get("models", {}).get("negbinom")
    if nb_model is None:
        log.error("Joblib no tiene artifact['models']['negbinom'] — reentrená antes")
        sys.exit(1)
    alpha = getattr(nb_model.family, "alpha", None)
    if alpha is None:
        log.warning("negbinom.family.alpha ausente → defecto statsmodels=1.0")
        return 1.0
    return float(alpha)


def _negbin_params(mu: float, alpha: float) -> tuple[float, float]:
    """Convierte (μ, α) → (n, p) para scipy.stats.nbinom.

    Parametrización scipy: X ~ NB(n, p) cuenta nro de fallos antes del n-ésimo éxito.
      mean = n(1-p)/p,  var = n(1-p)/p²
    Con n = 1/α y p = 1/(1+α·μ):
      mean = (1/α)·(1 - 1/(1+αμ))/(1/(1+αμ)) = (1/α)·(αμ) = μ          ✓
      var  = mean/p = μ·(1+αμ) = μ + α·μ²                               ✓
    """
    if alpha <= 0:
        raise ValueError(f"alpha debe ser > 0, recibido {alpha}")
    if mu <= 0:
        raise ValueError(f"mu debe ser > 0, recibido {mu}")
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    return n, p


def _verify_params(mu: float, alpha: float) -> None:
    """Sanity check: confirma que la conversión (μ,α) → (n,p) respeta mean y var."""
    n, p = _negbin_params(mu, alpha)
    got_mean = nbinom.mean(n, p)
    got_var = nbinom.var(n, p)
    expected_var = mu + alpha * mu**2
    assert abs(got_mean - mu) < 1e-6, f"nbinom.mean={got_mean} ≠ μ={mu}"
    assert abs(got_var - expected_var) < 1e-6, (
        f"nbinom.var={got_var} ≠ μ+αμ²={expected_var}"
    )


def _p_over_poisson(mu: float, line: float) -> float:
    """P(X > line) bajo Poisson(μ). Líneas .5 → floor(line)."""
    if mu <= 0:
        return 0.0
    return float(1.0 - poisson.cdf(int(line), mu))


def _p_over_negbin(mu: float, alpha: float, line: float) -> float:
    """P(X > line) bajo NegBin(μ, α). Mismo floor(line) que producción."""
    if mu <= 0:
        return 0.0
    n, p = _negbin_params(mu, alpha)
    return float(1.0 - nbinom.cdf(int(line), n, p))


def _load_predictions() -> pd.DataFrame:
    """Carga predictions y normaliza nombres de equipos para join con odds."""
    pred = pd.read_parquet(PREDICTIONS_PATH)
    pred = pred[pred["match_date"].astype(str).str.startswith(TARGET_DATE_PREFIX)].copy()
    if pred.empty:
        log.error("Sin predicciones para %s en %s", TARGET_DATE_PREFIX, PREDICTIONS_PATH)
        sys.exit(1)
    pred["home_ds"] = pred["home_team"].map(
        lambda s: _INVESTIGATION_ALIASES.get(_normalize_team_name(s), _normalize_team_name(s))
    )
    pred["away_ds"] = pred["away_team"].map(
        lambda s: _INVESTIGATION_ALIASES.get(_normalize_team_name(s), _normalize_team_name(s))
    )
    return pred[["home_team", "away_team", "home_ds", "away_ds", "pred_total_uniform"]]


def _load_odds() -> pd.DataFrame:
    """Carga odds 22bet del día, pivota over/under por (match, line)."""
    odds = pd.read_parquet(ODDS_PATH)
    odds = odds[
        (odds["match_date"].astype(str).str.startswith(TARGET_DATE_PREFIX))
        & (odds["market_type"] == "total_over_under")
    ].copy()
    odds["home_ds"] = odds["home_team"].map(
        lambda s: _INVESTIGATION_ALIASES.get(_normalize_team_name(s), _normalize_team_name(s))
    )
    odds["away_ds"] = odds["away_team"].map(
        lambda s: _INVESTIGATION_ALIASES.get(_normalize_team_name(s), _normalize_team_name(s))
    )
    # Nos quedamos con la cuota más reciente por (home, away, line, side).
    odds = odds.sort_values("scraped_at", ascending=False, kind="stable").drop_duplicates(
        ["home_ds", "away_ds", "line", "side"], keep="first"
    )
    wide = odds.pivot_table(
        index=["home_ds", "away_ds", "line"],
        columns="side",
        values="odds",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns={"over": "odds_over", "under": "odds_under"})
    wide = wide.dropna(subset=["odds_over", "odds_under"])
    return wide


def main() -> int:
    """Runner principal — genera CSV y log con veredicto."""
    if not PREDICTIONS_PATH.exists():
        log.error("Predicciones ausentes: %s", PREDICTIONS_PATH)
        return 1
    if not ODDS_PATH.exists():
        log.error("Odds ausentes: %s", ODDS_PATH)
        return 1

    alpha = _extract_alpha(MODEL_PATH)
    log.info("NegBinom fitted_alpha=%.6f (statsmodels default=1.0)", alpha)

    pred = _load_predictions()
    log.info("Predicciones del %s: %d partidos", TARGET_DATE_PREFIX, len(pred))

    odds = _load_odds()
    log.info("Odds 22bet O/U del %s: %d (match, line) pairs", TARGET_DATE_PREFIX, len(odds))

    merged = pred.merge(odds, on=["home_ds", "away_ds"], how="inner")
    if merged.empty:
        log.error("Merge vacío: no hay predicciones con odds el día %s", TARGET_DATE_PREFIX)
        return 1

    # Verificación de parametrización sobre el primer μ no-trivial.
    _verify_params(mu=float(merged["pred_total_uniform"].iloc[0]), alpha=alpha)
    log.info("Parametrización NB (n, p) verificada: mean y var matchean (μ,α).")

    rows: list[dict] = []
    for _, r in merged.iterrows():
        mu = float(r["pred_total_uniform"])
        line = float(r["line"])
        o_over = float(r["odds_over"])
        o_under = float(r["odds_under"])
        p_mkt_over, p_mkt_under = devig_proportional(o_over, o_under)

        p_pois_over = _p_over_poisson(mu, line)
        p_pois_under = 1.0 - p_pois_over
        p_nb_over = _p_over_negbin(mu, alpha, line)
        p_nb_under = 1.0 - p_nb_over

        edge_pois_over = p_pois_over - p_mkt_over
        edge_pois_under = p_pois_under - p_mkt_under
        edge_nb_over = p_nb_over - p_mkt_over
        edge_nb_under = p_nb_under - p_mkt_under

        match_label = f"{r['home_team']} vs {r['away_team']}"
        rows.append({
            "match": match_label,
            "side": "over",
            "line": line,
            "odds": o_over,
            "pred_mean": mu,
            "p_poisson": p_pois_over,
            "p_negbin": p_nb_over,
            "edge_poisson": edge_pois_over,
            "edge_negbin": edge_nb_over,
            "delta_edge": edge_nb_over - edge_pois_over,
        })
        rows.append({
            "match": match_label,
            "side": "under",
            "line": line,
            "odds": o_under,
            "pred_mean": mu,
            "p_poisson": p_pois_under,
            "p_negbin": p_nb_under,
            "edge_poisson": edge_pois_under,
            "edge_negbin": edge_nb_under,
            "delta_edge": edge_nb_under - edge_pois_under,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["match", "side", "line"], kind="stable").reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    log.info("CSV escrito: %s (%d filas)", OUTPUT_CSV, len(df))

    # ── Sociedad-Getafe focused table (UNDER) ────────────────────────────────
    sg_mask = df["match"].str.contains("Sociedad", case=False, na=False) & df["match"].str.contains(
        "Getafe", case=False, na=False
    )
    sg_under = df[sg_mask & (df["side"] == "under")].sort_values("line").reset_index(drop=True)
    print("\n" + "=" * 88)
    print("Sociedad vs Getafe — UNDER rows (mean-matched variance)")
    print("=" * 88)
    if sg_under.empty:
        print("(no rows — check match name normalization)")
    else:
        show = sg_under[["line", "odds", "p_poisson", "p_negbin", "edge_poisson", "edge_negbin", "delta_edge"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"{x:.4f}"))
        mean_delta = float(sg_under["delta_edge"].mean())
        median_delta = float(sg_under["delta_edge"].median())
        print(f"\nΔedge (UNDER, Sociedad-Getafe): mean={mean_delta:+.4f}, median={median_delta:+.4f}")

    # ── Aggregate stats across all matches ──────────────────────────────────
    print("\n" + "=" * 88)
    print("Aggregate Δedge stats (all matches, both sides)")
    print("=" * 88)
    for side_name, sub in df.groupby("side"):
        print(
            f"  side={side_name:>5}: n={len(sub):>3} mean={sub['delta_edge'].mean():+.4f} "
            f"median={sub['delta_edge'].median():+.4f} std={sub['delta_edge'].std():.4f} "
            f"min={sub['delta_edge'].min():+.4f} max={sub['delta_edge'].max():+.4f}"
        )

    # ── Verdict on Sociedad-Getafe UNDER ────────────────────────────────────
    if sg_under.empty:
        verdict = "NO_DATA"
    else:
        mean_delta = float(sg_under["delta_edge"].mean())
        if mean_delta < -0.10:
            verdict = "CONFIRMED"  # variance hypothesis confirmed
        elif abs(mean_delta) < 0.03:
            verdict = "STRUCTURAL"  # gap is not variance
        elif -0.10 <= mean_delta <= -0.03:
            verdict = "PARTIAL"
        elif mean_delta > 0:
            verdict = "UNEXPECTED"  # NB increases edges → probable bug
        else:
            verdict = "PARTIAL"

    headline_delta = float(sg_under["delta_edge"].abs().mean()) if not sg_under.empty else float("nan")
    log.info(
        "mean_matched test: fitted_alpha=%.6f; Sociedad-Getafe mean|Δedge|=%.4f; verdict=%s",
        alpha,
        headline_delta,
        verdict,
    )
    print(f"\nVerdict: {verdict}")
    print(f"fitted_alpha={alpha}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
