"""
Evaluate vs Market — Backtesting histórico de cuotas Codere
===========================================================
Cruza las predicciones del modelo con las cuotas históricas de Codere
para calcular métricas de rendimiento vs mercado, por mercado.

Mercados soportados:
  - total_over_under  (2 lados: over / under)
  - team_with_more    (3 lados: home / away / draw — Codere lista draw explícito)

Uso:
  python scripts/evaluation/evaluate_vs_market.py
  python scripts/evaluation/evaluate_vs_market.py --market team_with_more
  python scripts/evaluation/evaluate_vs_market.py --market all
  python scripts/evaluation/evaluate_vs_market.py --devig shin
  python scripts/evaluation/evaluate_vs_market.py --pred-col pred_total_v2

Outputs (bajo data/model/market_eval/):
  eval_total_over_under_YYYYMMDD.{parquet,csv}
  eval_team_with_more_YYYYMMDD.{parquet,csv}
  eval_summary_YYYYMMDD.json         — métricas agregadas por mercado
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap as scipy_bootstrap
from statsmodels.stats.proportion import proportion_confint

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
    p_home_more,
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

SUPPORTED_MARKETS = ("total_over_under", "team_with_more")
FALLBACK_WARN_THRESHOLD = 0.20   # ADR D4: WARN si >20% filas caen a rolling5
MC_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y CRUCE — TOTAL OVER/UNDER
# ─────────────────────────────────────────────────────────────────────────────

def load_odds_ou(devig_fn) -> pd.DataFrame:
    """Carga Codere O/U, aplica devig, devuelve una fila por partido con línea."""
    co = pd.read_parquet(ODDS_CODERE)
    co_ou = co[co["market_type"] == "total_over_under"].dropna(subset=["line"]).copy()

    co_ou["home_ds"] = co_ou["home_team"].map(lambda x: normalize_team(x, "codere_to_ds"))
    co_ou["away_ds"] = co_ou["away_team"].map(lambda x: normalize_team(x, "codere_to_ds"))

    over  = co_ou[co_ou["side"] == "over"].rename(columns={"odds": "odds_over"})
    under = co_ou[co_ou["side"] == "under"].rename(columns={"odds": "odds_under"})

    key_cols = ["home_ds", "away_ds", "line"]
    merged = over[key_cols + ["odds_over"]].merge(
        under[key_cols + ["odds_under"]], on=key_cols, how="inner"
    )

    merged[["p_mkt_over", "p_mkt_under"]] = merged.apply(
        lambda r: pd.Series(devig_fn(r["odds_over"], r["odds_under"])), axis=1
    )
    merged["vig_pct"] = merged.apply(
        lambda r: vig_pct(r["odds_over"], r["odds_under"]), axis=1
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y CRUCE — TEAM WITH MORE (3-way)
# ─────────────────────────────────────────────────────────────────────────────

def _devig_three_way_proportional(odds_home: float, odds_away: float, odds_draw: float) -> tuple[float, float, float]:
    """Devig proporcional sobre 3 selecciones (home/away/draw)."""
    r_h = 1.0 / odds_home
    r_a = 1.0 / odds_away
    r_d = 1.0 / odds_draw
    total = r_h + r_a + r_d
    return r_h / total, r_a / total, r_d / total


def _vig_pct_three_way(odds_home: float, odds_away: float, odds_draw: float) -> float:
    return (1.0 / odds_home + 1.0 / odds_away + 1.0 / odds_draw - 1.0) * 100.0


def load_odds_team_with_more() -> pd.DataFrame:
    """
    Carga Codere `team_with_more` (home/away/draw). Devuelve una fila por partido
    con columnas de odds y probabilidades devigged para las 3 selecciones.

    Si un match no trae las 3 selecciones (draw faltante por ejemplo), se descarta
    y se emite WARN — pricing 3-way requiere las tres cuotas.
    """
    co = pd.read_parquet(ODDS_CODERE)
    twm = co[co["market_type"] == "team_with_more"].copy()
    if twm.empty:
        return pd.DataFrame()

    twm["home_ds"] = twm["home_team"].map(lambda x: normalize_team(x, "codere_to_ds"))
    twm["away_ds"] = twm["away_team"].map(lambda x: normalize_team(x, "codere_to_ds"))

    # Snapshot más reciente por (partido, side) — evita que scrapes múltiples dupliquen
    twm = twm.sort_values("scraped_at", ascending=False, kind="stable")
    twm = twm.drop_duplicates(subset=["home_ds", "away_ds", "side"], keep="first")

    key_cols = ["home_ds", "away_ds"]
    h = twm[twm["side"] == "home"].rename(columns={"odds": "odds_home"})[key_cols + ["odds_home"]]
    a = twm[twm["side"] == "away"].rename(columns={"odds": "odds_away"})[key_cols + ["odds_away"]]
    d = twm[twm["side"] == "draw"].rename(columns={"odds": "odds_draw"})[key_cols + ["odds_draw"]]

    merged = h.merge(a, on=key_cols, how="inner").merge(d, on=key_cols, how="inner")
    n_incomplete = twm.groupby(["home_ds", "away_ds"]).size().lt(3).sum()
    if n_incomplete:
        log.warning(
            "team_with_more: %d partidos sin las 3 selecciones (home/away/draw) — descartados",
            int(n_incomplete),
        )

    if merged.empty:
        return merged

    probs = merged.apply(
        lambda r: pd.Series(
            _devig_three_way_proportional(r["odds_home"], r["odds_away"], r["odds_draw"]),
            index=["p_mkt_home", "p_mkt_away", "p_mkt_draw"],
        ),
        axis=1,
    )
    merged = pd.concat([merged, probs], axis=1)
    merged["vig_pct"] = merged.apply(
        lambda r: _vig_pct_three_way(r["odds_home"], r["odds_away"], r["odds_draw"]),
        axis=1,
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS REALES
# ─────────────────────────────────────────────────────────────────────────────

def load_results_ou() -> pd.DataFrame:
    """Resultados reales — totales por partido para O/U."""
    ds = pd.read_parquet(DATASET)
    ds26 = ds[ds["season"] == "2025/2026"]

    totals = ds26.groupby("match_id")["throw_ins_total"].sum().reset_index()
    totals.columns = ["match_id", "real_total"]

    home_info = ds26[ds26["is_home"] == 1][
        ["match_id", "team_name", "opponent_name", "match_date"]
    ].copy()
    home_info.columns = ["match_id", "home_ds", "away_ds", "match_date"]

    return totals.merge(home_info, on="match_id")


def load_results_team_with_more() -> pd.DataFrame:
    """
    Resultados reales para team_with_more: por partido, throw-ins del home y away
    separados + outcome categórico `realized_side ∈ {home, away, draw}`.
    """
    ds = pd.read_parquet(DATASET)
    ds26 = ds[ds["season"] == "2025/2026"]

    home_rows = ds26[ds26["is_home"] == 1][
        ["match_id", "match_date", "team_name", "opponent_name", "throw_ins_total"]
    ].rename(columns={
        "team_name": "home_ds",
        "opponent_name": "away_ds",
        "throw_ins_total": "real_home",
    })
    away_rows = ds26[ds26["is_home"] == 0][["match_id", "throw_ins_total"]].rename(
        columns={"throw_ins_total": "real_away"}
    )
    merged = home_rows.merge(away_rows, on="match_id", how="inner")

    def _outcome(h: float, a: float) -> str:
        if h > a:
            return "home"
        if a > h:
            return "away"
        return "draw"

    merged["realized_side"] = merged.apply(lambda r: _outcome(r["real_home"], r["real_away"]), axis=1)
    return merged


def cross_keys(odds_df: pd.DataFrame, results_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Merge por (home_ds, away_ds). Reporta partidos sin cruzar."""
    merged = results_df.merge(odds_df, on=["home_ds", "away_ds"], how="inner")
    odds_keys = set(zip(odds_df["home_ds"], odds_df["away_ds"]))
    res_keys  = set(zip(results_df["home_ds"], results_df["away_ds"]))
    unmatched = sorted(f"{h} vs {a}" for h, a in odds_keys - res_keys)
    return merged, unmatched


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIONES — carga desde parquet + data_source tagging (ADR D4)
# ─────────────────────────────────────────────────────────────────────────────

def _load_pred_file(pf: Path, cols: list[str]) -> pd.DataFrame | None:
    """Carga un parquet de predicciones con las columnas solicitadas (si existen)."""
    pred = pd.read_parquet(pf)
    missing = [c for c in cols if c not in pred.columns]
    if missing:
        return None
    pred = pred[["home_team", "away_team", "match_date", *cols]].copy()
    pred["home_ds"] = pred["home_team"].map(
        lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds")
    )
    pred["away_ds"] = pred["away_team"].map(
        lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds")
    )
    pred["match_date_pred"] = pd.to_datetime(pred["match_date"]).dt.date
    return pred


def attach_predictions_ou(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """
    Cruza predicciones (modelo v2) al backtest de O/U y añade `data_source`:
      - model_v2         → cruce directo por (home, away, fecha) con `pred_col`
      - rolling5_fallback → proxy desde dataset (rolling5 home + away)
      - unmatched        → no se pudo recuperar predicción

    Emite WARN si fallback >20% (ADR D4). Nunca hard-fail.
    """
    df = df.copy()
    df["pred_total"] = np.nan
    df["data_source"] = "unmatched"

    pred_dir = OUTPUT_DIR.parent / "predictions"
    pred_files = sorted(pred_dir.glob("predictions_*.parquet"))
    n_matched = 0

    for pf in pred_files:
        pred_slim = _load_pred_file(pf, [pred_col])
        if pred_slim is None:
            continue
        pred_slim = pred_slim.rename(columns={pred_col: "pred_total_tmp"})
        df_dates = pd.to_datetime(df["match_date"]).dt.date
        for idx in df.index:
            if df.at[idx, "data_source"] == "model_v2":
                continue
            row = df.loc[idx]
            match = pred_slim[
                (pred_slim["home_ds"] == row["home_ds"])
                & (pred_slim["away_ds"] == row["away_ds"])
                & (pred_slim["match_date_pred"] == df_dates.loc[idx])
            ]
            if not match.empty:
                df.at[idx, "pred_total"] = float(match["pred_total_tmp"].values[0])
                df.at[idx, "data_source"] = "model_v2"
                n_matched += 1

    if n_matched:
        log.info("Predicciones modelo_v2 cargadas: %d partidos", n_matched)

    # Fallback rolling5 — proxy
    missing_mask = df["data_source"] == "unmatched"
    n_missing = int(missing_mask.sum())
    if n_missing:
        log.info("Reconstruyendo λ proxy (rolling5) para %d partidos...", n_missing)
        ds = pd.read_parquet(DATASET)
        ds26 = ds[ds["season"] == "2025/2026"].copy()

        home_r5 = ds26[ds26["is_home"] == 1][
            ["match_id", "team_name", "opponent_name", "rolling5_throw_ins_total"]
        ].copy()
        away_r5 = ds26[ds26["is_home"] == 0][
            ["match_id", "team_name", "rolling5_throw_ins_total"]
        ].copy()

        r5 = home_r5.merge(away_r5, on="match_id", suffixes=("_home", "_away"))
        r5["lam_proxy"] = (
            r5["rolling5_throw_ins_total_home"].fillna(18)
            + r5["rolling5_throw_ins_total_away"].fillna(18)
        )
        r5["home_ds"] = r5["team_name_home"].map(
            lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds")
        )
        r5["away_ds"] = r5["opponent_name"].map(
            lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds")
        )
        proxy_map = dict(zip(zip(r5["home_ds"], r5["away_ds"]), r5["lam_proxy"]))

        for idx in df.index[missing_mask]:
            row = df.loc[idx]
            key = (row["home_ds"], row["away_ds"])
            if key in proxy_map and not pd.isna(proxy_map[key]):
                df.at[idx, "pred_total"] = float(proxy_map[key])
                df.at[idx, "data_source"] = "rolling5_fallback"

    counts = df["data_source"].value_counts().to_dict()
    n_total = len(df)
    frac_fallback = counts.get("rolling5_fallback", 0) / max(n_total, 1)
    log.info("data_source counts: %s", counts)
    if frac_fallback > FALLBACK_WARN_THRESHOLD:
        log.warning(
            "rolling5_fallback = %.1f%% de filas (>20%%). Métricas del modelo degradadas — revisar pipeline.",
            frac_fallback * 100,
        )

    still_nan = int(df["pred_total"].isna().sum())
    if still_nan:
        log.warning("%d partidos sin λ → excluidos de métricas de modelo", still_nan)
    return df


def attach_predictions_team_with_more(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza predicciones (modelo v2) al backtest de team_with_more:
    necesita `pred_home_v2` y `pred_away_v2` por separado.

    Tagging:
      - model_v2          → cruce directo
      - rolling5_fallback → rolling5 home / rolling5 away desde dataset
      - unmatched         → no recuperable
    """
    df = df.copy()
    df["pred_home_lam"] = np.nan
    df["pred_away_lam"] = np.nan
    df["data_source"] = "unmatched"

    pred_dir = OUTPUT_DIR.parent / "predictions"
    pred_files = sorted(pred_dir.glob("predictions_*.parquet"))
    n_matched = 0

    for pf in pred_files:
        pred_slim = _load_pred_file(pf, ["pred_home_v2", "pred_away_v2"])
        if pred_slim is None:
            continue
        df_dates = pd.to_datetime(df["match_date"]).dt.date
        for idx in df.index:
            if df.at[idx, "data_source"] == "model_v2":
                continue
            row = df.loc[idx]
            match = pred_slim[
                (pred_slim["home_ds"] == row["home_ds"])
                & (pred_slim["away_ds"] == row["away_ds"])
                & (pred_slim["match_date_pred"] == df_dates.loc[idx])
            ]
            if not match.empty:
                df.at[idx, "pred_home_lam"] = float(match["pred_home_v2"].values[0])
                df.at[idx, "pred_away_lam"] = float(match["pred_away_v2"].values[0])
                df.at[idx, "data_source"] = "model_v2"
                n_matched += 1

    if n_matched:
        log.info("Predicciones modelo_v2 cargadas (team_with_more): %d partidos", n_matched)

    missing_mask = df["data_source"] == "unmatched"
    n_missing = int(missing_mask.sum())
    if n_missing:
        log.info("Reconstruyendo λ proxy (rolling5 por lado) para %d partidos...", n_missing)
        ds = pd.read_parquet(DATASET)
        ds26 = ds[ds["season"] == "2025/2026"].copy()

        home_r5 = ds26[ds26["is_home"] == 1][
            ["match_id", "team_name", "opponent_name", "rolling5_throw_ins_total"]
        ].rename(columns={"rolling5_throw_ins_total": "lam_home_proxy"})
        away_r5 = ds26[ds26["is_home"] == 0][
            ["match_id", "rolling5_throw_ins_total"]
        ].rename(columns={"rolling5_throw_ins_total": "lam_away_proxy"})
        r5 = home_r5.merge(away_r5, on="match_id", how="inner")
        r5["lam_home_proxy"] = r5["lam_home_proxy"].fillna(18)
        r5["lam_away_proxy"] = r5["lam_away_proxy"].fillna(18)
        r5["home_ds"] = r5["team_name"].map(
            lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds")
        )
        r5["away_ds"] = r5["opponent_name"].map(
            lambda x: normalize_team(normalize_team(x, "ds_to_codere"), "codere_to_ds")
        )
        proxy_map = {
            (h, a): (lh, la)
            for h, a, lh, la in zip(
                r5["home_ds"], r5["away_ds"], r5["lam_home_proxy"], r5["lam_away_proxy"]
            )
        }
        for idx in df.index[missing_mask]:
            row = df.loc[idx]
            key = (row["home_ds"], row["away_ds"])
            if key in proxy_map:
                lh, la = proxy_map[key]
                if pd.notna(lh) and pd.notna(la):
                    df.at[idx, "pred_home_lam"] = float(lh)
                    df.at[idx, "pred_away_lam"] = float(la)
                    df.at[idx, "data_source"] = "rolling5_fallback"

    counts = df["data_source"].value_counts().to_dict()
    n_total = len(df)
    frac_fallback = counts.get("rolling5_fallback", 0) / max(n_total, 1)
    log.info("data_source counts (team_with_more): %s", counts)
    if frac_fallback > FALLBACK_WARN_THRESHOLD:
        log.warning(
            "rolling5_fallback = %.1f%% de filas en team_with_more (>20%%) — revisar pipeline.",
            frac_fallback * 100,
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS POR FILA
# ─────────────────────────────────────────────────────────────────────────────

def compute_rows_ou(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por partido para total_over_under."""
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

    df["ev_over"]  = df.apply(
        lambda r: expected_value(r["p_model_over"], r["odds_over"])
        if pd.notna(r["p_model_over"]) else np.nan,
        axis=1,
    )
    df["ev_under"] = df.apply(
        lambda r: expected_value(r["p_model_under"], r["odds_under"])
        if pd.notna(r["p_model_under"]) else np.nan,
        axis=1,
    )

    # side_picked: lado con mayor EV del modelo (tie → over)
    def _pick(row) -> str:
        if pd.isna(row["ev_over"]) or pd.isna(row["ev_under"]):
            return "none"
        return "over" if row["ev_over"] >= row["ev_under"] else "under"
    df["side_picked"] = df.apply(_pick, axis=1)

    df["market_type"] = "total_over_under"
    return df


def compute_rows_team_with_more(df: pd.DataFrame) -> pd.DataFrame:
    """
    Métricas por partido para team_with_more:
    usa Skellam (closed-form) → 3 probabilidades → EV por selección.
    """
    df = df.copy()
    has_pred = df["pred_home_lam"].notna() & df["pred_away_lam"].notna()

    p_home = np.full(len(df), np.nan)
    p_tie = np.full(len(df), np.nan)
    p_away = np.full(len(df), np.nan)

    if has_pred.any():
        lam_h = df.loc[has_pred, "pred_home_lam"].to_numpy(dtype=float)
        lam_a = df.loc[has_pred, "pred_away_lam"].to_numpy(dtype=float)
        p_h_v, p_t_v, p_a_v = p_home_more(lam_h, lam_a, method="skellam", seed=MC_SEED)
        p_home[has_pred.values] = np.asarray(p_h_v)
        p_tie[has_pred.values] = np.asarray(p_t_v)
        p_away[has_pred.values] = np.asarray(p_a_v)

    df["p_model_home"] = p_home
    df["p_model_draw"] = p_tie
    df["p_model_away"] = p_away

    df["edge_home"] = df["p_model_home"] - df["p_mkt_home"]
    df["edge_away"] = df["p_model_away"] - df["p_mkt_away"]
    df["edge_draw"] = df["p_model_draw"] - df["p_mkt_draw"]

    df["ev_home"] = df.apply(
        lambda r: expected_value(r["p_model_home"], r["odds_home"])
        if pd.notna(r["p_model_home"]) else np.nan,
        axis=1,
    )
    df["ev_away"] = df.apply(
        lambda r: expected_value(r["p_model_away"], r["odds_away"])
        if pd.notna(r["p_model_away"]) else np.nan,
        axis=1,
    )
    df["ev_draw"] = df.apply(
        lambda r: expected_value(r["p_model_draw"], r["odds_draw"])
        if pd.notna(r["p_model_draw"]) else np.nan,
        axis=1,
    )

    def _pick(row) -> str:
        evs = {"home": row["ev_home"], "away": row["ev_away"], "draw": row["ev_draw"]}
        evs = {k: v for k, v in evs.items() if pd.notna(v)}
        if not evs:
            return "none"
        return max(evs, key=evs.get)

    df["side_picked"] = df.apply(_pick, axis=1)
    df["market_type"] = "team_with_more"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS AGREGADAS
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(p_pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p_pred - y) ** 2))


def log_loss_binary(p_pred: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(p_pred, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    lo, hi = proportion_confint(count=k, nobs=n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def _roi_for_picks(df_m: pd.DataFrame, odds_col: str, win_mask: pd.Series) -> float:
    """ROI teórico flat-stake apostando siempre `side_picked`."""
    returns = np.where(win_mask, df_m[odds_col] - 1.0, -1.0)
    if len(returns) == 0:
        return float("nan")
    return float(np.mean(returns))


def aggregate_ou(df: pd.DataFrame) -> dict:
    """Métricas agregadas para total_over_under. Duales: `_model_v2` y `_all`."""
    y_full = df["realized_over"].values.astype(float)
    out: dict = {
        "market": "total_over_under",
        "n_all": int(len(df)),
        "realized_over_rate": float(y_full.mean()) if len(y_full) else float("nan"),
    }
    out["data_source_counts"] = {
        k: int(v) for k, v in df["data_source"].value_counts().items()
    }

    # Mercado (siempre sobre filas cruzadas)
    p_mkt = df["p_mkt_over"].values
    out["market_brier"] = brier_score(p_mkt, y_full) if len(y_full) else float("nan")
    out["market_log_loss"] = log_loss_binary(p_mkt, y_full) if len(y_full) else float("nan")
    out["avg_vig_pct"] = float(df["vig_pct"].mean()) if len(df) else float("nan")

    # Modelo — dos subconjuntos: model_v2 vs all
    for subset_name, mask in (("model_v2", df["data_source"] == "model_v2"),
                              ("all", df["pred_total"].notna())):
        sub = df[mask].copy()
        n = len(sub)
        if n == 0:
            out[f"n_{subset_name}"] = 0
            continue

        y = sub["realized_over"].values.astype(float)

        # hit_rate: fue el lado elegido el que ganó?
        wins = sub.apply(
            lambda r: (r["side_picked"] == "over" and r["realized_over"] == 1)
                      or (r["side_picked"] == "under" and r["realized_over"] == 0),
            axis=1,
        )
        k = int(wins.sum())
        lo, hi = _wilson_ci(k, n)

        # p_model para Brier/logloss sobre over (binario)
        p_mod = sub["p_model_over"].values

        # ROI teórico (y con push-refund, que para O/U 2-way sin push = mismo valor)
        def _ret(r):
            if r["side_picked"] == "over":
                return (r["odds_over"] - 1.0) if r["realized_over"] == 1 else -1.0
            return (r["odds_under"] - 1.0) if r["realized_over"] == 0 else -1.0
        returns = sub.apply(_ret, axis=1).astype(float).values
        roi = float(np.mean(returns)) if len(returns) else float("nan")

        out[f"n_{subset_name}"] = n
        out[f"hit_rate_{subset_name}"] = float(k / n)
        out[f"wilson_ci_low_{subset_name}"] = lo
        out[f"wilson_ci_high_{subset_name}"] = hi
        out[f"brier_{subset_name}"] = brier_score(p_mod, y)
        out[f"log_loss_{subset_name}"] = log_loss_binary(p_mod, y)
        out[f"roi_theoretical_{subset_name}"] = roi
        # Para O/U 2-way el push es P(X==line) — en líneas .5 = 0; asumimos igual a roi.
        out[f"roi_with_push_refund_{subset_name}"] = roi

    # Claves "principales" (sin sufijo) = subset model_v2 si existe, si no all
    primary = "model_v2" if out.get("n_model_v2", 0) > 0 else "all"
    if out.get(f"n_{primary}", 0) > 0:
        out["n"] = out[f"n_{primary}"]
        out["hit_rate"] = out[f"hit_rate_{primary}"]
        out["wilson_ci_low"] = out[f"wilson_ci_low_{primary}"]
        out["wilson_ci_high"] = out[f"wilson_ci_high_{primary}"]
        out["brier"] = out[f"brier_{primary}"]
        out["log_loss"] = out[f"log_loss_{primary}"]
        out["roi_theoretical"] = out[f"roi_theoretical_{primary}"]
        out["roi_with_push_refund"] = out[f"roi_with_push_refund_{primary}"]
    return out


def aggregate_team_with_more(df: pd.DataFrame) -> dict:
    """Métricas agregadas para team_with_more (3-way)."""
    out: dict = {
        "market": "team_with_more",
        "n_all": int(len(df)),
    }
    out["realized_side_counts"] = {
        k: int(v) for k, v in df["realized_side"].value_counts().items()
    }
    out["data_source_counts"] = {
        k: int(v) for k, v in df["data_source"].value_counts().items()
    }

    # VIG medio (3-way)
    out["avg_vig_pct"] = float(df["vig_pct"].mean()) if len(df) else float("nan")

    for subset_name, mask in (("model_v2", df["data_source"] == "model_v2"),
                              ("all", df["pred_home_lam"].notna())):
        sub = df[mask].copy()
        n = len(sub)
        if n == 0:
            out[f"n_{subset_name}"] = 0
            continue

        # hit_rate: side_picked == realized_side
        wins = (sub["side_picked"] == sub["realized_side"]).astype(int)
        k = int(wins.sum())
        lo, hi = _wilson_ci(k, n)

        # Brier / log_loss: evaluar sobre el outcome categórico como one-hot
        realized_home = (sub["realized_side"] == "home").astype(float).values
        realized_away = (sub["realized_side"] == "away").astype(float).values
        realized_draw = (sub["realized_side"] == "draw").astype(float).values

        p_h = sub["p_model_home"].values
        p_a = sub["p_model_away"].values
        p_d = sub["p_model_draw"].values

        # Multiclass Brier = media de los 3 Briers one-vs-rest (∈ [0, 2] convencional,
        # aquí promediamos los tres para mantener misma escala que binario).
        brier_multi = float(
            np.mean((p_h - realized_home) ** 2
                    + (p_a - realized_away) ** 2
                    + (p_d - realized_draw) ** 2)
        )
        # Log-loss multiclass: -mean(log(p_realized))
        eps = 1e-7
        p_realized = np.where(
            sub["realized_side"].values == "home", p_h,
            np.where(sub["realized_side"].values == "away", p_a, p_d),
        )
        p_realized = np.clip(p_realized, eps, 1 - eps)
        log_loss_multi = float(-np.mean(np.log(p_realized)))

        # ROI
        def _ret(r):
            side = r["side_picked"]
            realized = r["realized_side"]
            if side == "home":
                return (r["odds_home"] - 1.0) if realized == "home" else -1.0
            if side == "away":
                return (r["odds_away"] - 1.0) if realized == "away" else -1.0
            if side == "draw":
                return (r["odds_draw"] - 1.0) if realized == "draw" else -1.0
            return 0.0
        returns = sub.apply(_ret, axis=1).astype(float).values
        roi = float(np.mean(returns)) if len(returns) else float("nan")

        out[f"n_{subset_name}"] = n
        out[f"hit_rate_{subset_name}"] = float(k / n)
        out[f"wilson_ci_low_{subset_name}"] = lo
        out[f"wilson_ci_high_{subset_name}"] = hi
        out[f"brier_{subset_name}"] = brier_multi
        out[f"log_loss_{subset_name}"] = log_loss_multi
        out[f"roi_theoretical_{subset_name}"] = roi
        # 3-way con draw explícito → no hay push-refund (draw es un side propio). Igualamos.
        out[f"roi_with_push_refund_{subset_name}"] = roi

    primary = "model_v2" if out.get("n_model_v2", 0) > 0 else "all"
    if out.get(f"n_{primary}", 0) > 0:
        out["n"] = out[f"n_{primary}"]
        out["hit_rate"] = out[f"hit_rate_{primary}"]
        out["wilson_ci_low"] = out[f"wilson_ci_low_{primary}"]
        out["wilson_ci_high"] = out[f"wilson_ci_high_{primary}"]
        out["brier"] = out[f"brier_{primary}"]
        out["log_loss"] = out[f"log_loss_{primary}"]
        out["roi_theoretical"] = out[f"roi_theoretical_{primary}"]
        out["roi_with_push_refund"] = out[f"roi_with_push_refund_{primary}"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics: dict) -> None:
    market = metrics.get("market", "?")
    print("\n" + "=" * 72)
    print(f"MERCADO: {market}")
    print(f"  N total cruzado: {metrics.get('n_all', 0)}")
    print(f"  data_source: {metrics.get('data_source_counts', {})}")
    print(f"  VIG medio: {metrics.get('avg_vig_pct', float('nan')):.2f}%")
    print("=" * 72)
    if metrics.get("n", 0) == 0:
        print("  Sin predicciones disponibles — solo métricas mercado.")
        return
    print(f"  hit_rate:        {metrics['hit_rate']:.3f}  "
          f"(Wilson CI95: [{metrics['wilson_ci_low']:.3f}, {metrics['wilson_ci_high']:.3f}])  "
          f"N={metrics['n']}")
    print(f"  Brier:           {metrics['brier']:.4f}")
    print(f"  log-loss:        {metrics['log_loss']:.4f}")
    print(f"  ROI teórico:     {metrics['roi_theoretical']:+.4f}")
    if metrics.get("n_model_v2", 0) and metrics.get("n_all", 0) > metrics.get("n_model_v2", 0):
        print(f"  (subset model_v2 vs all visibles en eval_summary.json)")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE POR MERCADO
# ─────────────────────────────────────────────────────────────────────────────

def run_ou(devig_fn, pred_col: str) -> tuple[pd.DataFrame, dict]:
    odds = load_odds_ou(devig_fn)
    results = load_results_ou()
    log.info("O/U Codere: %d pares partido×línea", len(odds))
    log.info("Resultados reales 2025/26: %d partidos", results["match_id"].nunique())

    df, unmatched = cross_keys(odds, results)
    if unmatched:
        log.warning("O/U partidos en cuotas SIN resultado (unmatched): %d", len(unmatched))
    log.info("O/U partidos cruzados: %d", len(df))

    df = attach_predictions_ou(df, pred_col)
    df = compute_rows_ou(df)

    # Determinismo: ordenar por claves estables
    sort_cols = [c for c in ["match_date", "match_id", "home_ds", "away_ds", "line"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    metrics = aggregate_ou(df)
    return df, metrics


def run_team_with_more() -> tuple[pd.DataFrame, dict]:
    odds = load_odds_team_with_more()
    results = load_results_team_with_more()
    log.info("team_with_more Codere: %d partidos con 3 selecciones", len(odds))

    if odds.empty:
        log.warning("Sin cuotas team_with_more — skip")
        return pd.DataFrame(), {"market": "team_with_more", "n_all": 0}

    df, unmatched = cross_keys(odds, results)
    if unmatched:
        log.warning("team_with_more unmatched: %d", len(unmatched))
    log.info("team_with_more partidos cruzados: %d", len(df))

    df = attach_predictions_team_with_more(df)
    df = compute_rows_team_with_more(df)

    sort_cols = [c for c in ["match_date", "match_id", "home_ds", "away_ds"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    metrics = aggregate_team_with_more(df)
    return df, metrics


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas object mixtas a str para que parquet no explote."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            # si hay datetime.date, convertir a str
            out[col] = out[col].astype(object).where(out[col].notna(), None)
    return out


def save_market(df: pd.DataFrame, market: str, today_str: str) -> dict:
    if df.empty:
        return {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / f"eval_{market}_{today_str}"
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")

    df_clean = _clean_for_parquet(df)
    try:
        df_clean.to_parquet(parquet_path, index=False)
    except Exception as e:
        log.warning("No pude escribir parquet %s (%s). Uso solo CSV.", parquet_path, e)
        parquet_path = None
    df_clean.to_csv(csv_path, index=False)
    log.info("Guardado %s → %s%s", market, csv_path, f" + {parquet_path}" if parquet_path else "")
    return {"parquet": str(parquet_path) if parquet_path else None, "csv": str(csv_path)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Backtesting modelo vs mercado Codere (per-market)")
    parser.add_argument("--devig", choices=["proportional", "shin"], default="proportional",
                        help="Método de devig para O/U (default: proportional)")
    parser.add_argument("--pred-col", default="pred_total_v2",
                        help="Columna de predicción total para O/U (default: pred_total_v2)")
    parser.add_argument("--market", choices=["total_over_under", "team_with_more", "all"],
                        default="all",
                        help="Mercado a evaluar (default: all)")
    parser.add_argument("--output-dir", default=None,
                        help="Override del directorio de output")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    devig_fn = DEVIG_METHODS[args.devig]
    log.info("Devig=%s | pred_col=%s | market=%s", args.devig, args.pred_col, args.market)

    today_str = date.today().strftime("%Y%m%d")
    summary: dict = {
        "run_date": today_str,
        "devig": args.devig,
        "pred_col": args.pred_col,
        "markets": {},
        "artifacts": {},
    }

    markets_to_run = SUPPORTED_MARKETS if args.market == "all" else (args.market,)

    for market in markets_to_run:
        if market == "total_over_under":
            df, metrics = run_ou(devig_fn, args.pred_col)
        elif market == "team_with_more":
            df, metrics = run_team_with_more()
        else:
            continue

        print_summary(metrics)
        summary["markets"][market] = metrics
        paths = save_market(df, market, today_str)
        if paths:
            summary["artifacts"][market] = paths

    # Summary JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"eval_summary_{today_str}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str, sort_keys=True)
    log.info("Summary JSON: %s", summary_path)


if __name__ == "__main__":
    main()
