"""
Value Bets — Detección de apuestas con valor (live)
====================================================
Carga las predicciones del día + las cuotas más recientes scrapeadas
(Codere y/o 22bet) y filtra aquellas con Expected Value positivo
por encima de un umbral configurable.

Mercados:
  - total_over_under  (2-way: over/under)
  - team_with_more    (3-way: home/away/draw — Codere lista draw explícito)

Uso:
  python scripts/evaluation/value_bets.py
  python scripts/evaluation/value_bets.py --market team_with_more
  python scripts/evaluation/value_bets.py --market all
  python scripts/evaluation/value_bets.py --ev-threshold 0.08 --edge-min 0.04
  python scripts/evaluation/value_bets.py --no-save   # solo stdout

Output:
  data/model/market_eval/value_bets_YYYYMMDD.csv

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
    TEAM_NAME_MAP,
    devig_proportional,
    expected_value,
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
) -> pd.DataFrame:
    """Cruza predicciones O/U con cuotas y detecta value bets (over/under)."""
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
        }

        for side, ev, edge, p_model, p_mkt, odds_val in [
            ("over",  ev_over,  edge_over,  p_model_over,  p_mkt_over,  o_odds),
            ("under", ev_under, edge_under, p_model_under, p_mkt_under, u_odds),
        ]:
            if ev >= ev_threshold and abs(edge) >= edge_min:
                rows.append({
                    **base,
                    "side": side,
                    "odds": round(odds_val, 2),
                    "p_model": round(p_model, 3),
                    "p_market": round(p_mkt, 3),
                    "edge": round(edge, 3),
                    "ev": round(ev, 4),
                })

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

        specs = [
            ("home",  float(r["p_model_home"]), p_mkt_h, odds_h),
            ("away",  float(r["p_model_away"]), p_mkt_a, odds_a),
            ("draw",  float(r["p_model_draw"]), p_mkt_d, odds_d),
        ]
        for side, p_model, p_mkt, odds_val in specs:
            # EV 3-way: EV = p_model * odds - 1 (sin push-refund, draw es selección propia).
            ev = expected_value(p_model, odds_val)
            edge = p_model - p_mkt
            if ev >= ev_threshold and abs(edge) >= edge_min:
                rows.append({
                    **base,
                    "side": side,
                    "odds": round(float(odds_val), 2),
                    "p_model": round(p_model, 3),
                    "p_market": round(p_mkt, 3),
                    "edge": round(edge, 3),
                    "ev": round(ev, 4),
                })

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
    parser.add_argument("--no-save", action="store_true",
                        help="Solo stdout, no guardar CSV")
    args = parser.parse_args()

    markets = ("total_over_under", "team_with_more") if args.market == "all" else (args.market,)
    all_rows: list[pd.DataFrame] = []

    if "total_over_under" in markets:
        pred_total = load_latest_predictions_total(args.pred_col)
        log.info("Partidos O/U a evaluar: %d", len(pred_total))
        odds_ou = _combine_non_empty(load_codere_odds_ou(), load_22bet_odds_ou())
        if odds_ou is None:
            log.warning("Sin cuotas O/U disponibles — skip")
        else:
            vb_ou = compute_value_bets_ou(pred_total, odds_ou, args.ev_threshold, args.edge_min)
            if not vb_ou.empty:
                all_rows.append(vb_ou)

    if "team_with_more" in markets:
        pred_pair = load_latest_predictions_per_team()
        log.info("Partidos team_with_more a evaluar: %d", len(pred_pair))
        odds_twm = load_codere_odds_team_with_more()
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
