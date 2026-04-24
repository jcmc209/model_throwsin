"""
Value Bets — Detección de apuestas con valor (live)
====================================================
Carga las predicciones del día + las cuotas más recientes scrapeadas
(Codere y/o 22bet) y filtra aquellas con Expected Value positivo
por encima de un umbral configurable.

Mercados:
  - total_over_under  (2-way: over/under)
  - team_with_more    (3-way: home/away/draw — Codere y 22bet listan draw
    explícito → sin push-refund)

Line shopping (cross-book):
  Con `--book all` (default), para cada selección cruzada con predicciones se
  elige la MEJOR CUOTA entre Codere y 22bet y se reporta `best_price_book` en
  el CSV de salida. Esto maximiza el EV esperado sin cambiar el modelo.

Uso:
  python scripts/evaluation/value_bets.py
  python scripts/evaluation/value_bets.py --market team_with_more
  python scripts/evaluation/value_bets.py --market all
  python scripts/evaluation/value_bets.py --book codere       # un solo libro
  python scripts/evaluation/value_bets.py --book 22bet
  python scripts/evaluation/value_bets.py --book all          # line shopping
  python scripts/evaluation/value_bets.py --variance-model negbin   # default
  python scripts/evaluation/value_bets.py --variance-model poisson  # legacy
  python scripts/evaluation/value_bets.py --alpha 0.01         # override α NegBin
  python scripts/evaluation/value_bets.py --ev-threshold 0.08 --edge-min 0.04
  python scripts/evaluation/value_bets.py --no-save   # solo stdout

Probabilidad O/U:
  `--variance-model=negbin` (default) usa X ~ NegBin(μ, α) con α =
  DEFAULT_NEGBIN_ALPHA (tuneado offline vía `scripts/investigation/
  tune_alpha_via_cv.py` sobre val 2025/2026). Corrige la underdispersion
  estructural del Poisson en saques de banda. `--variance-model=poisson`
  preserva el path histórico para diff/comparación.

Output:
  data/model/market_eval/value_bets_YYYYMMDD.csv
  Con columna `best_price_book` ∈ {codere, 22bet} cuando --book=all.

Interpretación EV:
  EV > 0.05  → devuelve 5%+ sobre la apuesta en expectativa
  EV > 0.10  → señal más fuerte (raro con 9% de vig)
  EV < 0     → la casa tiene ventaja → no apostar
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.market_utils import (
    DEFAULT_NEGBIN_ALPHA,
    TEAM_NAME_MAP,
    devig_proportional,
    expected_value,
    nbinom_over_prob,
    nbinom_under_prob,
    normalize_team,
    p_home_more,
    poisson_over_prob,
    poisson_under_prob,
    vig_pct,
)
from scripts.evaluation._normalize import _normalize_team_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("value_bets")

PREDICTIONS_DIR = _root / "data/model/predictions"
ODDS_CODERE     = _root / "data/reference/odds_codere.parquet"
ODDS_22BET      = _root / "data/reference/odds_22bet.parquet"
OUTPUT_DIR      = _root / "data/model/market_eval"

# Umbrales por defecto — conservadores con N pequeña
DEFAULT_EV_THRESHOLD   = 0.05
DEFAULT_EDGE_MIN       = 0.03
DEFAULT_PRED_COL       = "pred_total_v2"
MC_SEED                = 42


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE PREDICCIONES
# ─────────────────────────────────────────────────────────────────────────────

def _latest_predictions_file() -> Path:
    files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
    if not files:
        log.error(
            "No hay archivos de predicciones en %s. Ejecuta: python -m model.predict --matchday next",
            PREDICTIONS_DIR,
        )
        sys.exit(1)
    return files[-1]


def load_latest_predictions_total(pred_col: str) -> pd.DataFrame:
    """Predicciones para mercado O/U: λ_total por partido."""
    pred_file = _latest_predictions_file()
    log.info("Predicciones: %s", pred_file.name)
    pred = pd.read_parquet(pred_file)

    if pred_col not in pred.columns:
        log.error("Columna '%s' no encontrada. Disponibles: %s", pred_col, list(pred.columns))
        sys.exit(1)

    pred = pred[["home_team", "away_team", "match_date", pred_col]].copy()
    pred.columns = ["home_ds", "away_ds", "match_date", "pred_total"]
    pred["home_ds"] = pred["home_ds"].map(_normalize_team_name)
    pred["away_ds"] = pred["away_ds"].map(_normalize_team_name)
    return pred


def load_latest_predictions_per_team() -> pd.DataFrame:
    """Predicciones para team_with_more: λ_home y λ_away por partido."""
    pred_file = _latest_predictions_file()
    pred = pd.read_parquet(pred_file)
    missing = [c for c in ("pred_home_v2", "pred_away_v2") if c not in pred.columns]
    if missing:
        log.error("Columnas %s no encontradas en %s — se requieren para team_with_more",
                  missing, pred_file.name)
        sys.exit(1)

    pred = pred[["home_team", "away_team", "match_date", "pred_home_v2", "pred_away_v2"]].copy()
    pred = pred.rename(columns={
        "home_team": "home_ds",
        "away_team": "away_ds",
        "pred_home_v2": "pred_home_lam",
        "pred_away_v2": "pred_away_lam",
    })
    pred["home_ds"] = pred["home_ds"].map(_normalize_team_name)
    pred["away_ds"] = pred["away_ds"].map(_normalize_team_name)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE ODDS
# ─────────────────────────────────────────────────────────────────────────────

def load_codere_odds_ou() -> pd.DataFrame | None:
    """Codere O/U — snapshot más reciente por partido+línea."""
    if not ODDS_CODERE.exists():
        log.warning("odds_codere.parquet no encontrado.")
        return None

    co = pd.read_parquet(ODDS_CODERE)
    co_ou = co[co["market_type"] == "total_over_under"].dropna(subset=["line"]).copy()
    if co_ou.empty:
        return None

    co_ou["home_ds"] = co_ou["home_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )
    co_ou["away_ds"] = co_ou["away_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )

    co_ou = co_ou.sort_values("scraped_at", ascending=False, kind="stable")
    co_ou = co_ou.drop_duplicates(subset=["home_ds", "away_ds", "line", "side"], keep="first")

    over  = co_ou[co_ou["side"] == "over"].rename(columns={"odds": "odds_over"})[
        ["home_ds", "away_ds", "line", "odds_over", "scraped_at"]]
    under = co_ou[co_ou["side"] == "under"].rename(columns={"odds": "odds_under"})[
        ["home_ds", "away_ds", "line", "odds_under"]]

    merged = over.merge(under, on=["home_ds", "away_ds", "line"], how="inner")
    merged["bookmaker"] = "codere"
    merged["market_type"] = "total_over_under"
    return merged


def load_22bet_odds_ou() -> pd.DataFrame | None:
    """22bet O/U — snapshot más reciente por partido+línea."""
    if not ODDS_22BET.exists():
        return None

    b = pd.read_parquet(ODDS_22BET)
    over  = b[b["side"] == "over"].copy()
    under = b[b["side"] == "under"].copy()
    if over.empty or under.empty:
        return None

    time_col = "scraped_at" if "scraped_at" in b.columns else None
    if time_col:
        over  = over.sort_values(time_col, ascending=False, kind="stable").drop_duplicates(
            ["home_team", "away_team", "line"])
        under = under.sort_values(time_col, ascending=False, kind="stable").drop_duplicates(
            ["home_team", "away_team", "line"])

    key = ["home_team", "away_team", "line"]
    merged = over[key + ["odds"]].rename(columns={"odds": "odds_over"}).merge(
        under[key + ["odds"]].rename(columns={"odds": "odds_under"}), on=key
    )
    merged["home_ds"] = merged["home_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )
    merged["away_ds"] = merged["away_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )
    merged["bookmaker"] = "22bet"
    merged["scraped_at"] = over["scraped_at"].values[0] if time_col and len(over) else None
    merged["market_type"] = "total_over_under"
    return merged[["home_ds", "away_ds", "line", "odds_over", "odds_under",
                   "bookmaker", "scraped_at", "market_type"]]


def load_codere_odds_team_with_more() -> pd.DataFrame | None:
    """Codere team_with_more — pivota home/away/draw en la misma fila."""
    if not ODDS_CODERE.exists():
        return None

    co = pd.read_parquet(ODDS_CODERE)
    twm = co[co["market_type"] == "team_with_more"].copy()
    if twm.empty:
        return None

    twm["home_ds"] = twm["home_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )
    twm["away_ds"] = twm["away_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )

    twm = twm.sort_values("scraped_at", ascending=False, kind="stable")
    twm = twm.drop_duplicates(subset=["home_ds", "away_ds", "side"], keep="first")

    key = ["home_ds", "away_ds"]
    h = twm[twm["side"] == "home"].rename(columns={"odds": "odds_home"})[key + ["odds_home", "scraped_at"]]
    a = twm[twm["side"] == "away"].rename(columns={"odds": "odds_away"})[key + ["odds_away"]]
    d = twm[twm["side"] == "draw"].rename(columns={"odds": "odds_draw"})[key + ["odds_draw"]]

    merged = h.merge(a, on=key, how="inner").merge(d, on=key, how="inner")
    if merged.empty:
        return None
    merged["bookmaker"] = "codere"
    merged["market_type"] = "team_with_more"
    return merged


def load_22bet_odds_team_with_more() -> pd.DataFrame | None:
    """22bet team_with_more — pivota home/away/draw en la misma fila.

    Mirror del loader Codere. Requiere que el parquet de 22bet tenga la columna
    `market_type` (añadida por el scraper v2). Si es un parquet legacy sin esa
    columna, devuelve None.
    """
    if not ODDS_22BET.exists():
        return None

    b = pd.read_parquet(ODDS_22BET)
    if "market_type" not in b.columns:
        log.warning("odds_22bet.parquet sin columna market_type — scraper legacy. "
                    "Re-ejecuta `python scripts/odds/22bet_scraper.py` para refrescar.")
        return None

    twm = b[b["market_type"] == "team_with_more"].copy()
    if twm.empty:
        return None

    twm["home_ds"] = twm["home_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )
    twm["away_ds"] = twm["away_team"].map(
        lambda x: _normalize_team_name(normalize_team(x, "codere_to_ds"))
    )

    # scraped_at puede venir como str en parquets de 22bet
    if twm["scraped_at"].dtype == object:
        twm["scraped_at"] = pd.to_datetime(twm["scraped_at"], utc=True, errors="coerce")

    twm = twm.sort_values("scraped_at", ascending=False, kind="stable")
    twm = twm.drop_duplicates(subset=["home_ds", "away_ds", "side"], keep="first")

    key = ["home_ds", "away_ds"]
    h = twm[twm["side"] == "home"].rename(columns={"odds": "odds_home"})[key + ["odds_home", "scraped_at"]]
    a = twm[twm["side"] == "away"].rename(columns={"odds": "odds_away"})[key + ["odds_away"]]
    d = twm[twm["side"] == "draw"].rename(columns={"odds": "odds_draw"})[key + ["odds_draw"]]

    merged = h.merge(a, on=key, how="inner").merge(d, on=key, how="inner")
    if merged.empty:
        return None
    merged["bookmaker"] = "22bet"
    merged["market_type"] = "team_with_more"
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# LINE SHOPPING — mejor precio por selección cruzando libros
# ─────────────────────────────────────────────────────────────────────────────

def _best_price_ou(*dfs: pd.DataFrame | None) -> pd.DataFrame | None:
    """Cruza O/U de varios libros y se queda con la MEJOR CUOTA por lado y
    línea. Para `over` y `under` se maximiza independientemente la cuota →
    el resultado es un par (best_over, best_under) que puede venir de libros
    distintos (a propósito: es line shopping puro).

    Añade dos columnas de traza:
      - `best_book_over`: libro que aporta `odds_over`.
      - `best_book_under`: libro que aporta `odds_under`.
    """
    valid = [d for d in dfs if d is not None and len(d) > 0]
    if not valid:
        return None

    full = pd.concat(valid, ignore_index=True)

    # Elegimos best odds_over por grupo (home_ds, away_ds, line)
    over = (full.sort_values("odds_over", ascending=False, kind="stable")
                .drop_duplicates(["home_ds", "away_ds", "line"], keep="first")
                [["home_ds", "away_ds", "line", "odds_over", "bookmaker", "scraped_at"]]
                .rename(columns={"bookmaker": "best_book_over"}))

    under = (full.sort_values("odds_under", ascending=False, kind="stable")
                 .drop_duplicates(["home_ds", "away_ds", "line"], keep="first")
                 [["home_ds", "away_ds", "line", "odds_under", "bookmaker"]]
                 .rename(columns={"bookmaker": "best_book_under"}))

    merged = over.merge(under, on=["home_ds", "away_ds", "line"], how="inner")
    merged["market_type"] = "total_over_under"
    # bookmaker legacy: marcamos como 'best' para compat con downstream
    merged["bookmaker"] = "best"
    return merged


def _best_price_twm(*dfs: pd.DataFrame | None) -> pd.DataFrame | None:
    """Cruza team_with_more de varios libros y se queda con la MEJOR CUOTA por
    cada lado (home, draw, away) independientemente.

    Añade columnas:
      - `best_book_home`, `best_book_draw`, `best_book_away`.
    """
    valid = [d for d in dfs if d is not None and len(d) > 0]
    if not valid:
        return None

    full = pd.concat(valid, ignore_index=True)
    key = ["home_ds", "away_ds"]

    h = (full.sort_values("odds_home", ascending=False, kind="stable")
             .drop_duplicates(key, keep="first")
             [key + ["odds_home", "bookmaker", "scraped_at"]]
             .rename(columns={"bookmaker": "best_book_home"}))
    a = (full.sort_values("odds_away", ascending=False, kind="stable")
             .drop_duplicates(key, keep="first")
             [key + ["odds_away", "bookmaker"]]
             .rename(columns={"bookmaker": "best_book_away"}))
    d = (full.sort_values("odds_draw", ascending=False, kind="stable")
             .drop_duplicates(key, keep="first")
             [key + ["odds_draw", "bookmaker"]]
             .rename(columns={"bookmaker": "best_book_draw"}))

    merged = h.merge(a, on=key, how="inner").merge(d, on=key, how="inner")
    merged["market_type"] = "team_with_more"
    merged["bookmaker"] = "best"
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# DEVIG 3-way
# ─────────────────────────────────────────────────────────────────────────────

def _devig_three_way(odds_h: float, odds_a: float, odds_d: float) -> tuple[float, float, float]:
    r_h = 1.0 / odds_h
    r_a = 1.0 / odds_a
    r_d = 1.0 / odds_d
    t = r_h + r_a + r_d
    return r_h / t, r_a / t, r_d / t


def _vig_pct_three_way(odds_h: float, odds_a: float, odds_d: float) -> float:
    return (1.0 / odds_h + 1.0 / odds_a + 1.0 / odds_d - 1.0) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# CÓMPUTO DE VALUE BETS POR MERCADO
# ─────────────────────────────────────────────────────────────────────────────

def compute_value_bets_ou(
    pred: pd.DataFrame,
    odds: pd.DataFrame,
    ev_threshold: float,
    edge_min: float,
    variance_model: str = "negbin",
    alpha: float | None = None,
) -> pd.DataFrame:
    """Cruza predicciones O/U con cuotas y detecta value bets (over/under).

    Args:
        pred, odds, ev_threshold, edge_min: ver firma previa.
        variance_model: "negbin" (default, recomendado) | "poisson".
            NegBin usa α = `alpha` o `DEFAULT_NEGBIN_ALPHA` (tuned offline).
        alpha: override opcional para α NegBin. Ignorado si variance_model="poisson".
    """
    if variance_model not in ("poisson", "negbin"):
        raise ValueError(f"variance_model debe ser 'poisson' o 'negbin', recibí {variance_model!r}")
    alpha_eff = float(alpha) if alpha is not None else DEFAULT_NEGBIN_ALPHA

    merged = pred.merge(odds, on=["home_ds", "away_ds"], how="inner")
    if merged.empty:
        return merged

    rows = []
    for _, r in merged.iterrows():
        lam = r["pred_total"]
        line = r["line"]
        o_odds = r["odds_over"]
        u_odds = r["odds_under"]

        p_mkt_over, p_mkt_under = devig_proportional(o_odds, u_odds)
        if variance_model == "negbin":
            p_model_over  = float(nbinom_over_prob(lam, line, alpha_eff))
            p_model_under = float(nbinom_under_prob(lam, line, alpha_eff))
        else:
            p_model_over  = poisson_over_prob(lam, line)
            p_model_under = poisson_under_prob(lam, line)

        ev_over  = expected_value(p_model_over,  o_odds)
        ev_under = expected_value(p_model_under, u_odds)
        edge_over  = p_model_over  - p_mkt_over
        edge_under = p_model_under - p_mkt_under
        vig = vig_pct(o_odds, u_odds)

        base = {
            "market_type": "total_over_under",
            "home_team": r["home_ds"],
            "away_team": r["away_ds"],
            "match_date": r.get("match_date", ""),
            "line": line,
            "pred_total": round(lam, 2),
            "vig_pct": round(vig, 1),
            "bookmaker": r.get("bookmaker", "?"),
            "scraped_at": r.get("scraped_at", ""),
            "variance_model": variance_model,
            "alpha": round(alpha_eff, 4) if variance_model == "negbin" else None,
        }

        # Line shopping: si vinieron columnas best_book_*, propagamos el libro
        # concreto que ofreció la cuota ganadora por selección.
        best_book_over  = r.get("best_book_over")
        best_book_under = r.get("best_book_under")

        for side, ev, edge, p_model, p_mkt, odds_val, book_col in [
            ("over",  ev_over,  edge_over,  p_model_over,  p_mkt_over,  o_odds, best_book_over),
            ("under", ev_under, edge_under, p_model_under, p_mkt_under, u_odds, best_book_under),
        ]:
            if ev >= ev_threshold and abs(edge) >= edge_min:
                row = {
                    **base,
                    "side": side,
                    "odds": round(odds_val, 2),
                    "p_model": round(p_model, 3),
                    "p_market": round(p_mkt, 3),
                    "edge": round(edge, 3),
                    "ev": round(ev, 4),
                }
                if book_col is not None and not (isinstance(book_col, float) and pd.isna(book_col)):
                    row["best_price_book"] = book_col
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_value_bets_team_with_more(
    pred: pd.DataFrame,
    odds: pd.DataFrame,
    ev_threshold: float,
    edge_min: float,
) -> pd.DataFrame:
    """
    Cruza predicciones por-lado con cuotas team_with_more (3-way: home/away/draw).

    El mercado de Codere lista `draw` como selección explícita → NO hay push-refund.
    Cada lado se cotiza por separado y se computa EV estándar contra su cuota.
    """
    merged = pred.merge(odds, on=["home_ds", "away_ds"], how="inner")
    if merged.empty:
        return merged

    # Pricing vectorizado
    lam_h = merged["pred_home_lam"].to_numpy(dtype=float)
    lam_a = merged["pred_away_lam"].to_numpy(dtype=float)
    p_h, p_d, p_a = p_home_more(lam_h, lam_a, method="skellam", seed=MC_SEED)
    p_h = np.asarray(p_h, dtype=float)
    p_d = np.asarray(p_d, dtype=float)
    p_a = np.asarray(p_a, dtype=float)

    merged = merged.assign(
        p_model_home=p_h, p_model_draw=p_d, p_model_away=p_a,
    ).reset_index(drop=True)

    rows = []
    for i, r in merged.iterrows():
        odds_h = r["odds_home"]
        odds_a = r["odds_away"]
        odds_d = r["odds_draw"]
        p_mkt_h, p_mkt_a, p_mkt_d = _devig_three_way(odds_h, odds_a, odds_d)
        vig = _vig_pct_three_way(odds_h, odds_a, odds_d)

        base = {
            "market_type": "team_with_more",
            "home_team": r["home_ds"],
            "away_team": r["away_ds"],
            "match_date": r.get("match_date", ""),
            "line": None,  # team_with_more no tiene línea
            "pred_home_lam": round(float(r["pred_home_lam"]), 2),
            "pred_away_lam": round(float(r["pred_away_lam"]), 2),
            "vig_pct": round(vig, 1),
            "bookmaker": r.get("bookmaker", "?"),
            "scraped_at": r.get("scraped_at", ""),
        }

        # Line shopping 3-way: mejor libro por cada selección (puede variar).
        best_book_h = r.get("best_book_home")
        best_book_a = r.get("best_book_away")
        best_book_d = r.get("best_book_draw")

        specs = [
            ("home",  float(r["p_model_home"]), p_mkt_h, odds_h, best_book_h),
            ("away",  float(r["p_model_away"]), p_mkt_a, odds_a, best_book_a),
            ("draw",  float(r["p_model_draw"]), p_mkt_d, odds_d, best_book_d),
        ]
        for side, p_model, p_mkt, odds_val, book_col in specs:
            # EV 3-way: EV = p_model * odds - 1 (sin push-refund, draw es selección propia).
            ev = expected_value(p_model, odds_val)
            edge = p_model - p_mkt
            if ev >= ev_threshold and abs(edge) >= edge_min:
                row = {
                    **base,
                    "side": side,
                    "odds": round(float(odds_val), 2),
                    "p_model": round(p_model, 3),
                    "p_market": round(p_mkt, 3),
                    "edge": round(edge, 3),
                    "ev": round(ev, 4),
                }
                if book_col is not None and not (isinstance(book_col, float) and pd.isna(book_col)):
                    row["best_price_book"] = book_col
                rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_value_bets(df: pd.DataFrame, ev_threshold: float, edge_min: float) -> None:
    print("\n" + "=" * 100)
    print(f"VALUE BETS — {date.today()}  (EV ≥ {ev_threshold:.0%}, |edge| ≥ {edge_min:.0%})")
    print("=" * 100)

    if df.empty:
        print("  Sin value bets con los umbrales actuales.")
    else:
        pd.set_option("display.max_rows", 50)
        pd.set_option("display.width", 140)
        pd.set_option("display.float_format", "{:.3f}".format)

        display_cols = ["market_type", "home_team", "away_team", "line", "side", "odds",
                        "p_model", "p_market", "edge", "ev", "vig_pct", "bookmaker"]
        available = [c for c in display_cols if c in df.columns]
        print(df[available].to_string(index=False))
        print()
        print(f"  Total value bets encontradas: {len(df)}")
        print(f"  EV medio:   {df['ev'].mean():+.4f}")
        print(f"  Edge medio: {df['edge'].mean():+.4f}")

    print()
    print("  ⚠️  EV estimado con modelo; no validado con N estadísticamente suficiente.")
    print("     Usar Kelly fraccional (¼ Kelly). Nunca apostar flat sin gestión.")
    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _combine_non_empty(*dfs: pd.DataFrame | None) -> pd.DataFrame | None:
    valid = [d for d in dfs if d is not None and len(d) > 0]
    if not valid:
        return None
    return pd.concat(valid, ignore_index=True)


def _select_odds_ou(book: str) -> pd.DataFrame | None:
    """Devuelve cuotas O/U según el flag --book.

    book='codere' | '22bet' → cuotas de un solo libro (comportamiento histórico).
    book='all'              → best price por lado entre Codere y 22bet
                              (line shopping); añade columnas `best_book_over`
                              y `best_book_under`.
    """
    if book == "codere":
        return load_codere_odds_ou()
    if book == "22bet":
        return load_22bet_odds_ou()
    # book == 'all' → line shopping
    return _best_price_ou(load_codere_odds_ou(), load_22bet_odds_ou())


def _select_odds_twm(book: str) -> pd.DataFrame | None:
    """Devuelve cuotas team_with_more según el flag --book (mirror de _select_odds_ou)."""
    if book == "codere":
        return load_codere_odds_team_with_more()
    if book == "22bet":
        return load_22bet_odds_team_with_more()
    return _best_price_twm(
        load_codere_odds_team_with_more(),
        load_22bet_odds_team_with_more(),
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Detectar value bets en cuotas del día")
    parser.add_argument("--ev-threshold", type=float, default=DEFAULT_EV_THRESHOLD,
                        help=f"EV mínimo para reportar (default: {DEFAULT_EV_THRESHOLD})")
    parser.add_argument("--edge-min", type=float, default=DEFAULT_EDGE_MIN,
                        help=f"|edge| mínimo p_model−p_market (default: {DEFAULT_EDGE_MIN})")
    parser.add_argument("--pred-col", default=DEFAULT_PRED_COL,
                        help=f"Columna de predicción total (default: {DEFAULT_PRED_COL})")
    parser.add_argument("--market", choices=["total_over_under", "team_with_more", "all"],
                        default="total_over_under",
                        help="Mercado a analizar (default: total_over_under)")
    parser.add_argument("--book", choices=["codere", "22bet", "all"],
                        default="all",
                        help="Libro(s) de cuotas. 'all' = line shopping "
                             "(mejor precio por selección entre Codere y 22bet)")
    parser.add_argument("--variance-model", choices=["poisson", "negbin"],
                        default="negbin",
                        help="Distribución asumida para P(over/under). Default: negbin "
                             "(α = DEFAULT_NEGBIN_ALPHA de market_utils, ajustado offline).")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Dispersión NegBin. Default: constante DEFAULT_NEGBIN_ALPHA "
                             "de model/market_utils.py. Solo aplica si --variance-model=negbin.")
    parser.add_argument("--no-save", action="store_true",
                        help="Solo stdout, no guardar CSV")
    args = parser.parse_args()

    markets = ("total_over_under", "team_with_more") if args.market == "all" else (args.market,)
    all_rows: list[pd.DataFrame] = []

    alpha_effective = args.alpha if args.alpha is not None else DEFAULT_NEGBIN_ALPHA
    log.info(
        "Configuración: market=%s book=%s variance_model=%s%s",
        args.market, args.book, args.variance_model,
        f" alpha={alpha_effective:.4f}" if args.variance_model == "negbin" else "",
    )

    if "total_over_under" in markets:
        pred_total = load_latest_predictions_total(args.pred_col)
        log.info("Partidos O/U a evaluar: %d", len(pred_total))
        odds_ou = _select_odds_ou(args.book)
        if odds_ou is None:
            log.warning("Sin cuotas O/U disponibles — skip")
        else:
            vb_ou = compute_value_bets_ou(
                pred_total, odds_ou, args.ev_threshold, args.edge_min,
                variance_model=args.variance_model, alpha=args.alpha,
            )
            if not vb_ou.empty:
                all_rows.append(vb_ou)

    if "team_with_more" in markets:
        pred_pair = load_latest_predictions_per_team()
        log.info("Partidos team_with_more a evaluar: %d", len(pred_pair))
        odds_twm = _select_odds_twm(args.book)
        if odds_twm is None:
            log.warning("Sin cuotas team_with_more disponibles — skip")
        else:
            vb_twm = compute_value_bets_team_with_more(
                pred_pair, odds_twm, args.ev_threshold, args.edge_min
            )
            if not vb_twm.empty:
                all_rows.append(vb_twm)

    if all_rows:
        vb = pd.concat(all_rows, ignore_index=True)
        # Determinismo: orden por keys estables, luego por ev descendente
        sort_cols = [c for c in ["market_type", "home_team", "away_team", "side"] if c in vb.columns]
        vb = vb.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        # Re-ordenar para display por ev
        vb_display = vb.sort_values("ev", ascending=False, kind="stable").reset_index(drop=True)
    else:
        vb = pd.DataFrame()
        vb_display = vb

    print_value_bets(vb_display, args.ev_threshold, args.edge_min)

    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        today = date.today().strftime("%Y%m%d")
        csv_path = OUTPUT_DIR / f"value_bets_{today}.csv"
        if vb.empty:
            pd.DataFrame().to_csv(csv_path, index=False)
        else:
            vb.to_csv(csv_path, index=False)
        log.info("Guardado: %s", csv_path)


if __name__ == "__main__":
    main()
