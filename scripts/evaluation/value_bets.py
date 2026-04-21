"""
Value Bets — Detección de apuestas con valor (live)
====================================================
Carga las predicciones del día + las cuotas más recientes scrapeadas
(Codere y/o 22bet) y filtra aquellas con Expected Value positivo
por encima de un umbral configurable.

Uso:
  python scripts/evaluation/value_bets.py
  python scripts/evaluation/value_bets.py --ev-threshold 0.08
  python scripts/evaluation/value_bets.py --edge-min 0.04
  python scripts/evaluation/value_bets.py --pred-col pred_total_v2
  python scripts/evaluation/value_bets.py --no-save   # solo stdout

Output:
  data/model/market_eval/value_bets_YYYYMMDD.csv
  stdout: tabla rankeada por EV

Interpretación EV:
  EV > 0.05  → devuelve 5%+ sobre la apuesta en expectativa
  EV > 0.10  → señal más fuerte (raro con 9% de vig)
  EV < 0     → la casa tiene ventaja → no apostar

⚠️  Con N histórica pequeña (≈30 partidos) no podemos validar que el EV
    declarado sea real. Tratar como señal informativa, no como certeza.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
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
    poisson_over_prob,
    poisson_under_prob,
    vig_pct,
)

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
DEFAULT_EV_THRESHOLD   = 0.05   # mínimo EV para reportar (5%)
DEFAULT_EDGE_MIN       = 0.03   # mínimo |p_model − p_market| (3pp)
DEFAULT_PRED_COL       = "pred_total_v2"


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_predictions(pred_col: str) -> pd.DataFrame:
    """Carga el archivo de predicciones más reciente."""
    files = sorted(PREDICTIONS_DIR.glob("predictions_*.parquet"))
    if not files:
        log.error("No hay archivos de predicciones en %s. Ejecuta primero: python -m model.predict --matchday next", PREDICTIONS_DIR)
        sys.exit(1)
    pred_file = files[-1]
    log.info("Predicciones: %s", pred_file.name)
    pred = pd.read_parquet(pred_file)

    if pred_col not in pred.columns:
        log.error("Columna '%s' no encontrada. Disponibles: %s", pred_col, list(pred.columns))
        sys.exit(1)

    pred = pred[["home_team", "away_team", "match_date", pred_col]].copy()
    pred.columns = ["home_ds", "away_ds", "match_date", "pred_total"]

    # Normalizar nombres dataset → Codere → dataset (para asegurar consistencia)
    pred["home_ds"] = pred["home_ds"].map(lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds"))
    pred["away_ds"] = pred["away_ds"].map(lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds"))
    return pred


def load_codere_odds() -> pd.DataFrame | None:
    """Carga cuotas Codere O/U, quedándose con el snapshot más reciente por partido."""
    if not ODDS_CODERE.exists():
        log.warning("odds_codere.parquet no encontrado.")
        return None

    co = pd.read_parquet(ODDS_CODERE)
    co_ou = co[co["market_type"] == "total_over_under"].dropna(subset=["line"]).copy()
    if co_ou.empty:
        return None

    co_ou["home_ds"] = co_ou["home_team"].map(lambda x: normalize_team(x, "codere_to_ds"))
    co_ou["away_ds"] = co_ou["away_team"].map(lambda x: normalize_team(x, "codere_to_ds"))

    # Snapshot más reciente por partido+línea
    co_ou = co_ou.sort_values("scraped_at", ascending=False)
    co_ou = co_ou.drop_duplicates(subset=["home_ds", "away_ds", "line", "side"])

    over  = co_ou[co_ou["side"] == "over"].rename(columns={"odds": "odds_over"})[
        ["home_ds", "away_ds", "line", "odds_over", "scraped_at"]]
    under = co_ou[co_ou["side"] == "under"].rename(columns={"odds": "odds_under"})[
        ["home_ds", "away_ds", "line", "odds_under"]]

    merged = over.merge(under, on=["home_ds", "away_ds", "line"], how="inner")
    merged["bookmaker"] = "codere"
    return merged


def load_22bet_odds() -> pd.DataFrame | None:
    """Carga cuotas 22bet O/U, snapshot más reciente por partido+línea."""
    if not ODDS_22BET.exists():
        return None

    b = pd.read_parquet(ODDS_22BET)
    # 22bet tiene múltiples líneas por partido
    over  = b[b["side"] == "over"].copy()
    under = b[b["side"] == "under"].copy()
    if over.empty or under.empty:
        return None

    # Columna de tiempo
    time_col = "scraped_at" if "scraped_at" in b.columns else None
    if time_col:
        over  = over.sort_values(time_col, ascending=False).drop_duplicates(["home_team", "away_team", "line"])
        under = under.sort_values(time_col, ascending=False).drop_duplicates(["home_team", "away_team", "line"])

    key = ["home_team", "away_team", "line"]
    merged = over[key + ["odds"]].rename(columns={"odds": "odds_over"}).merge(
        under[key + ["odds"]].rename(columns={"odds": "odds_under"}), on=key
    )
    merged["home_ds"] = merged["home_team"].map(lambda x: normalize_team(x, "codere_to_ds"))
    merged["away_ds"] = merged["away_team"].map(lambda x: normalize_team(x, "codere_to_ds"))
    merged["bookmaker"] = "22bet"
    merged["scraped_at"] = over["scraped_at"].values[0] if time_col and len(over) else None

    return merged[["home_ds", "away_ds", "line", "odds_over", "odds_under", "bookmaker", "scraped_at"]]


def combine_odds(*dfs) -> pd.DataFrame:
    """Combina cuotas de varios bookmakers. Line shopping: para la misma línea, el mejor precio."""
    valid = [d for d in dfs if d is not None and len(d) > 0]
    if not valid:
        log.error("No hay cuotas disponibles de ningún bookmaker.")
        sys.exit(1)
    return pd.concat(valid, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE VALUE BETS
# ─────────────────────────────────────────────────────────────────────────────

def compute_value_bets(
    pred: pd.DataFrame,
    odds: pd.DataFrame,
    ev_threshold: float,
    edge_min: float,
) -> pd.DataFrame:
    """
    Cruza predicciones con cuotas y calcula EV para over y under de cada línea.
    Devuelve las filas con value (EV ≥ threshold y |edge| ≥ edge_min).
    """
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
    return pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_value_bets(df: pd.DataFrame, ev_threshold: float, edge_min: float) -> None:
    print("\n" + "=" * 100)
    print(f"VALUE BETS — {date.today()}  (EV ≥ {ev_threshold:.0%}, |edge| ≥ {edge_min:.0%})")
    print("=" * 100)

    if df.empty:
        print("  Sin value bets con los umbrales actuales.")
        print(f"  (Si el mercado tiene vig ~9% y N=29, un EV ≥ 5% real es poco frecuente.)")
    else:
        pd.set_option("display.max_rows", 50)
        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", "{:.3f}".format)

        display_cols = ["home_team", "away_team", "line", "side", "odds",
                        "pred_total", "p_model", "p_market", "edge", "ev",
                        "vig_pct", "bookmaker"]
        available = [c for c in display_cols if c in df.columns]
        print(df[available].to_string(index=True))
        print()
        print(f"  Total value bets encontradas: {len(df)}")
        print(f"  EV medio:   {df['ev'].mean():+.4f}")
        print(f"  Edge medio: {df['edge'].mean():+.4f}")

    print()
    print("  ⚠️  RECORDATORIO:")
    print("     • EV estimado con modelo; no validado con N estadísticamente suficiente (N<100).")
    print("     • Usar Kelly fraccional (¼ Kelly) para staking. Nunca apostar flat sin gestión.")
    print("     • El VIG del 9.3% requiere p_model ≥ p_mkt + 9pp para tener EV positivo neto real.")
    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Detectar value bets en cuotas del día")
    parser.add_argument("--ev-threshold", type=float, default=DEFAULT_EV_THRESHOLD,
                        help=f"EV mínimo para reportar (default: {DEFAULT_EV_THRESHOLD})")
    parser.add_argument("--edge-min", type=float, default=DEFAULT_EDGE_MIN,
                        help=f"|edge| mínimo p_model−p_market (default: {DEFAULT_EDGE_MIN})")
    parser.add_argument("--pred-col", default=DEFAULT_PRED_COL,
                        help=f"Columna de predicción (default: {DEFAULT_PRED_COL})")
    parser.add_argument("--no-save", action="store_true",
                        help="Solo stdout, no guardar CSV")
    args = parser.parse_args()

    # 1. Predicciones
    pred = load_latest_predictions(args.pred_col)
    log.info("Partidos a predecir: %d", len(pred))

    # 2. Cuotas (line shopping entre bookmakers)
    co_odds = load_codere_odds()
    b22_odds = load_22bet_odds()

    n_co  = len(co_odds)  if co_odds  is not None else 0
    n_b22 = len(b22_odds) if b22_odds is not None else 0
    log.info("Cuotas cargadas — Codere: %d líneas, 22bet: %d líneas", n_co, n_b22)

    all_odds = combine_odds(co_odds, b22_odds)

    # 3. Calcular value bets
    vb = compute_value_bets(pred, all_odds, args.ev_threshold, args.edge_min)

    # 4. Display
    print_value_bets(vb, args.ev_threshold, args.edge_min)

    # 5. Guardar
    if not args.no_save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        today = date.today().strftime("%Y%m%d")
        csv_path = OUTPUT_DIR / f"value_bets_{today}.csv"
        if vb.empty:
            # Guardar vacío igualmente para que quede registro
            pd.DataFrame().to_csv(csv_path, index=False)
        else:
            vb.to_csv(csv_path, index=False)
        log.info("Guardado: %s", csv_path)


if __name__ == "__main__":
    main()
