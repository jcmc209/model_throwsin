"""
Evaluate vs Market — Backtesting histórico de cuotas Codere
===========================================================
Cruza las predicciones del modelo con las cuotas históricas de Codere
para calcular métricas de rendimiento vs mercado.

Usa los 29 partidos de Codere (mar–abr 2026) que ya tienen resultado real.

Uso:
  python scripts/evaluation/evaluate_vs_market.py
  python scripts/evaluation/evaluate_vs_market.py --devig shin
  python scripts/evaluation/evaluate_vs_market.py --pred-col pred_total_v2
  python scripts/evaluation/evaluate_vs_market.py --output data/model/market_eval/eval_custom.parquet

Outputs:
  data/model/market_eval/eval_YYYYMMDD.parquet   — tabla por partido
  data/model/market_eval/eval_YYYYMMDD.csv       — versión legible
  (métricas agregadas se imprimen en stdout)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap as scipy_bootstrap

# ── path setup ────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.market_utils import (
    TEAM_NAME_MAP,
    devig_proportional,
    devig_shin,
    expected_value,
    normalize_team,
    poisson_over_prob,
    vig_pct,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("evaluate_vs_market")

# ── paths ─────────────────────────────────────────────────────
ODDS_CODERE = _root / "data/reference/odds_codere.parquet"
ODDS_22BET  = _root / "data/reference/odds_22bet.parquet"
DATASET     = _root / "data/model/dataset.parquet"
OUTPUT_DIR  = _root / "data/model/market_eval"

DEVIG_METHODS = {
    "proportional": devig_proportional,
    "shin": devig_shin,
}


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y CRUCE
# ─────────────────────────────────────────────────────────────────────────────

def load_odds_ou(devig_fn) -> pd.DataFrame:
    """Carga Codere O/U, aplica devig, devuelve una fila por partido con línea."""
    co = pd.read_parquet(ODDS_CODERE)
    co_ou = co[co["market_type"] == "total_over_under"].dropna(subset=["line"]).copy()

    # Normalizar nombres → WhoScored
    co_ou["home_ds"] = co_ou["home_team"].map(lambda x: normalize_team(x, "codere_to_ds"))
    co_ou["away_ds"] = co_ou["away_team"].map(lambda x: normalize_team(x, "codere_to_ds"))

    # Pivotear over/under en la misma fila
    over  = co_ou[co_ou["side"] == "over"].rename(columns={"odds": "odds_over"})
    under = co_ou[co_ou["side"] == "under"].rename(columns={"odds": "odds_under"})

    key_cols = ["home_ds", "away_ds", "line"]
    merged = over[key_cols + ["odds_over"]].merge(
        under[key_cols + ["odds_under"]], on=key_cols, how="inner"
    )

    # Devig
    merged[["p_mkt_over", "p_mkt_under"]] = merged.apply(
        lambda r: pd.Series(devig_fn(r["odds_over"], r["odds_under"])), axis=1
    )
    merged["vig_pct"] = merged.apply(
        lambda r: vig_pct(r["odds_over"], r["odds_under"]), axis=1
    )
    return merged


def load_results() -> pd.DataFrame:
    """Carga resultados reales del dataset (totales por partido)."""
    ds = pd.read_parquet(DATASET)
    ds26 = ds[ds["season"] == "2025/2026"]

    totals = ds26.groupby("match_id")["throw_ins_total"].sum().reset_index()
    totals.columns = ["match_id", "real_total"]

    home_info = ds26[ds26["is_home"] == 1][["match_id", "team_name", "opponent_name", "match_date"]].copy()
    home_info.columns = ["match_id", "home_ds", "away_ds", "match_date"]

    return totals.merge(home_info, on="match_id")


def cross_odds_results(odds_df: pd.DataFrame, results_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Cruza cuotas con resultados. Reporta partidos sin cruzar."""
    # Asegurar que match_date viaja en el merge aunque odds_df no lo tenga
    odds_merge = odds_df.copy()
    merged = results_df.merge(odds_merge, on=["home_ds", "away_ds"], how="inner")

    # Partidos en cuotas sin resultado
    odds_keys = set(zip(odds_df["home_ds"], odds_df["away_ds"]))
    res_keys  = set(zip(results_df["home_ds"], results_df["away_ds"]))
    unmatched = [f"{h} vs {a}" for h, a in odds_keys - res_keys]

    return merged, unmatched


def rebuild_predictions(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """
    Intenta reconstruir λ_total para los partidos históricos cruzados.

    Estrategia:
    1. Busca archivos de predicciones guardados y cruza por (home, away, match_date).
       Si el archivo es de hoy y los partidos son del pasado, el cruce no dará matches.
    2. Fallback: re-predice desde el dataset usando rolling5 + bias V2 por partido.
       No es idéntico al modelo real (que usa todos los features), pero es un proxy razonable.

    La forma óptima es ejecutar `python -m model.predict --date YYYY-MM-DD` para cada
    partido histórico y guardar las predicciones antes de correr este script.
    """
    # ── Intento 1: buscar en archivos de predicciones ─────────────────────
    pred_files = sorted(OUTPUT_DIR.parent.joinpath("predictions").glob("predictions_*.parquet"))
    n_matched = 0
    df["pred_total"] = np.nan

    for pf in pred_files:
        pred = pd.read_parquet(pf)
        if pred_col not in pred.columns:
            continue
        pred_slim = pred[["home_team", "away_team", "match_date", pred_col]].copy()
        pred_slim.columns = ["home_ds", "away_ds", "match_date_pred", "pred_total_tmp"]
        # Normalizar
        pred_slim["home_ds"] = pred_slim["home_ds"].map(
            lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds"))
        pred_slim["away_ds"] = pred_slim["away_ds"].map(
            lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds"))
        pred_slim["match_date_pred"] = pd.to_datetime(pred_slim["match_date_pred"]).dt.date

        if "match_date" in df.columns:
            df_dates = pd.to_datetime(df["match_date"]).dt.date
            pred_slim["match_date_pred"] = pd.to_datetime(pred_slim["match_date_pred"]).dt.date
            for idx in df.index:
                row = df.loc[idx]
                match = pred_slim[
                    (pred_slim["home_ds"] == row["home_ds"]) &
                    (pred_slim["away_ds"] == row["away_ds"]) &
                    (pred_slim["match_date_pred"] == df_dates.loc[idx])
                ]
                if not match.empty:
                    df.at[idx, "pred_total"] = match["pred_total_tmp"].values[0]
                    n_matched += 1

    if n_matched > 0:
        log.info("Predicciones del modelo cargadas desde archivos: %d partidos", n_matched)

    # ── Intento 2: proxy desde dataset (rolling5_throw_ins_total home+away) ─
    n_still_missing = df["pred_total"].isna().sum()
    if n_still_missing > 0:
        log.info("Reconstruyendo λ proxy para %d partidos desde rolling5 del dataset...", n_still_missing)
        ds = pd.read_parquet(DATASET)
        ds26 = ds[ds["season"] == "2025/2026"].copy()

        # rolling5 home+away por match_id
        home_r5 = ds26[ds26["is_home"] == 1][["match_id", "team_name", "opponent_name",
                                               "rolling5_throw_ins_total"]].copy()
        away_r5 = ds26[ds26["is_home"] == 0][["match_id", "team_name", "rolling5_throw_ins_total"]].copy()

        # Merge por match_id para obtener par
        r5 = home_r5.merge(away_r5, on="match_id", suffixes=("_home", "_away"))
        r5["lam_proxy"] = r5["rolling5_throw_ins_total_home"].fillna(18) + r5["rolling5_throw_ins_total_away"].fillna(18)
        r5["home_ds"] = r5["team_name_home"].map(lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds"))
        r5["away_ds"] = r5["opponent_name"].map(lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds"))

        proxy_map = dict(zip(zip(r5["home_ds"], r5["away_ds"]), r5["lam_proxy"]))

        mask = df["pred_total"].isna()
        df.loc[mask, "pred_total"] = df.loc[mask].apply(
            lambda row: proxy_map.get((row["home_ds"], row["away_ds"]), np.nan), axis=1
        )
        n_proxy = df["pred_total"].notna().sum() - (n_matched)
        log.info("Partidos con λ proxy (rolling5): %d", n_proxy)

    still_nan = df["pred_total"].isna().sum()
    if still_nan:
        log.warning("%d partidos sin λ → excluidos de métricas de modelo.", still_nan)
    else:
        log.info("λ disponible para todos los %d partidos cruzados.", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CÁLCULO DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Añade p_model_over, edge, ev_over, realized_over a nivel partido."""
    df = df.copy()
    df["realized_over"] = (df["real_total"] > df["line"]).astype(int)

    has_pred = df["pred_total"].notna()
    df["p_model_over"] = np.nan
    df.loc[has_pred, "p_model_over"] = df.loc[has_pred].apply(
        lambda r: poisson_over_prob(r["pred_total"], r["line"]), axis=1
    )
    df["p_model_under"] = 1.0 - df["p_model_over"]

    df["edge_over"]  = df["p_model_over"]  - df["p_mkt_over"]
    df["edge_under"] = df["p_model_under"] - df["p_mkt_under"]

    df["ev_over"]  = df.apply(lambda r: expected_value(r["p_model_over"],  r["odds_over"])  if pd.notna(r["p_model_over"])  else np.nan, axis=1)
    df["ev_under"] = df.apply(lambda r: expected_value(r["p_model_under"], r["odds_under"]) if pd.notna(r["p_model_under"]) else np.nan, axis=1)

    return df


def brier_score(p_pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p_pred - y) ** 2))


def log_loss_binary(p_pred: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def bootstrap_ci(metric_fn, data: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """IC bootstrap de percentil para una métrica escalar."""
    result = scipy_bootstrap(
        (data,), metric_fn, n_resamples=n_boot, confidence_level=ci, method="percentile"
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas agregadas sobre todos los partidos cruzados."""
    y = df["realized_over"].values.astype(float)
    n = len(y)
    metrics: dict = {"n_matches": n}

    # ── Naive baseline ────────────────────────────────────────
    metrics["naive_brier"]    = brier_score(np.full(n, 0.5), y)
    metrics["naive_logloss"]  = log_loss_binary(np.full(n, 0.5), y)
    metrics["realized_over_rate"] = float(y.mean())

    # ── Mercado ───────────────────────────────────────────────
    p_mkt = df["p_mkt_over"].values
    metrics["market_brier"]   = brier_score(p_mkt, y)
    metrics["market_logloss"] = log_loss_binary(p_mkt, y)
    metrics["market_accuracy"] = float(((p_mkt > 0.5) == y).mean())
    metrics["avg_vig_pct"]    = float(df["vig_pct"].mean())

    # ── Modelo ────────────────────────────────────────────────
    df_m = df.dropna(subset=["p_model_over"])
    if len(df_m) > 0:
        p_mod = df_m["p_model_over"].values
        y_m   = df_m["realized_over"].values.astype(float)
        metrics["model_brier"]    = brier_score(p_mod, y_m)
        metrics["model_logloss"]  = log_loss_binary(p_mod, y_m)
        metrics["model_accuracy"] = float(((p_mod > 0.5) == y_m).mean())
        metrics["model_n"]        = len(df_m)

        # ROI teórico: apostamos siempre el lado con mayor EV del modelo
        roi_returns = []
        for _, row in df_m.iterrows():
            if row["ev_over"] > row["ev_under"]:
                ret = row["odds_over"] - 1 if row["realized_over"] == 1 else -1
            else:
                ret = row["odds_under"] - 1 if row["realized_over"] == 0 else -1
            roi_returns.append(ret)
        metrics["roi_theoretical"] = float(np.mean(roi_returns))

        # IC bootstrap del Brier Score del modelo (advertir sobre N pequeña)
        if len(df_m) >= 10:
            # scipy.bootstrap espera una función (data,) → scalar, donde data es 1D resample
            # Pasamos p_mod e y_m como dos arrays separados
            def _brier_boot(p, y):
                return brier_score(p, y)
            result = scipy_bootstrap(
                (p_mod, y_m), _brier_boot,
                n_resamples=2000, confidence_level=0.95,
                method="percentile", paired=True,
            )
            metrics["model_brier_ci95"] = [
                float(result.confidence_interval.low),
                float(result.confidence_interval.high),
            ]
    else:
        log.warning("No hay predicciones de modelo disponibles. Solo se calculan métricas de mercado.")

    return metrics


def calibration_table(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Tabla de calibración: p_model_over en quintiles vs tasa real de over."""
    df_m = df.dropna(subset=["p_model_over"]).copy()
    if len(df_m) < n_bins:
        return pd.DataFrame()
    df_m["bin"] = pd.qcut(df_m["p_model_over"], q=n_bins, duplicates="drop")
    cal = df_m.groupby("bin", observed=True).agg(
        n=("realized_over", "count"),
        p_model_mean=("p_model_over", "mean"),
        realized_rate=("realized_over", "mean"),
    ).reset_index()
    cal["calibration_error"] = cal["p_model_mean"] - cal["realized_rate"]
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_match_table(df: pd.DataFrame) -> None:
    display_cols = ["home_ds", "away_ds", "line", "real_total", "realized_over",
                    "p_mkt_over", "p_model_over", "edge_over", "ev_over", "vig_pct"]
    available = [c for c in display_cols if c in df.columns]
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n" + "=" * 90)
    print("TABLA POR PARTIDO")
    print("=" * 90)
    sort_col = next((c for c in ["match_date", "home_ds"] if c in df.columns), None)
    out = df[available]
    if sort_col and sort_col in out.columns:
        out = out.sort_values(sort_col)
    print(out.to_string(index=False))


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("MÉTRICAS AGREGADAS")
    print(f"  ⚠️  N = {metrics['n_matches']} partidos — intervalo de confianza muy amplio")
    print("=" * 60)
    print(f"  Realized over rate:    {metrics['realized_over_rate']:.3f}  (esperado ≈0.50)")
    print(f"  VIG promedio:          {metrics.get('avg_vig_pct', 0):.1f}%")
    print()
    print("  BRIER SCORE (↓ mejor, naive=0.25):")
    print(f"    Naive (0.5):         {metrics['naive_brier']:.4f}")
    print(f"    Mercado:             {metrics['market_brier']:.4f}")
    if "model_brier" in metrics:
        ci = metrics.get("model_brier_ci95", [None, None])
        ci_str = f"  IC95%: [{ci[0]:.4f}, {ci[1]:.4f}]" if ci[0] is not None else ""
        print(f"    Modelo:              {metrics['model_brier']:.4f}{ci_str}")
    print()
    print("  LOG-LOSS (↓ mejor):")
    print(f"    Naive:               {metrics['naive_logloss']:.4f}")
    print(f"    Mercado:             {metrics['market_logloss']:.4f}")
    if "model_logloss" in metrics:
        print(f"    Modelo:              {metrics['model_logloss']:.4f}")
    print()
    print("  ACCURACY (lado correcto del 50%):")
    print(f"    Mercado:             {metrics['market_accuracy']:.3f}")
    if "model_accuracy" in metrics:
        print(f"    Modelo:              {metrics['model_accuracy']:.3f}")
    if "roi_theoretical" in metrics:
        print()
        print(f"  ROI TEÓRICO (siempre apostar lado con mayor EV del modelo):")
        print(f"    {metrics['roi_theoretical']:+.3f}  ({metrics['roi_theoretical']*100:+.1f}% por apuesta)")
        print("    ⚠️  Con N pequeña, ROI teórico es ruido. No usar para decisiones reales.")
    print("=" * 60)


def print_calibration(cal: pd.DataFrame) -> None:
    if cal.empty:
        return
    print("\nCALIBRACIÓN (p_model_over en quintiles):")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(cal.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Backtesting modelo vs mercado Codere")
    parser.add_argument("--devig", choices=["proportional", "shin"], default="proportional",
                        help="Método de devig (default: proportional)")
    parser.add_argument("--pred-col", default="pred_total_v2",
                        help="Columna de predicción a usar (default: pred_total_v2)")
    parser.add_argument("--output", default=None,
                        help="Ruta custom para el parquet de salida")
    args = parser.parse_args()

    devig_fn = DEVIG_METHODS[args.devig]
    log.info("Devig: %s | Columna predicción: %s", args.devig, args.pred_col)

    # 1. Cargar datos
    odds = load_odds_ou(devig_fn)
    results = load_results()
    log.info("Cuotas O/U Codere: %d pares partido×línea", len(odds))
    log.info("Resultados reales 2025/26: %d partidos", results["match_id"].nunique())

    # 2. Cruzar
    df, unmatched = cross_odds_results(odds, results)
    if unmatched:
        log.warning("Partidos en cuotas SIN resultado real (¿futuros o nombre distinto?):")
        for m in unmatched:
            log.warning("  → %s", m)
    log.info("Partidos cruzados con resultado: %d", len(df))

    # 3. Añadir predicciones del modelo
    df = rebuild_predictions(df, args.pred_col)

    # 4. Calcular métricas por partido
    df = compute_row_metrics(df)

    # 5. Display
    print_match_table(df)
    metrics = compute_aggregate_metrics(df)
    print_metrics(metrics)
    cal = calibration_table(df)
    print_calibration(cal)

    # 6. Guardar
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"eval_{today}.parquet"
    df.to_parquet(out_path, index=False)
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    log.info("Output guardado: %s", out_path)
    log.info("CSV legible:     %s", csv_path)


if __name__ == "__main__":
    main()
