"""
Dataset Builder — Throw-In Predictor
====================================
Construye el dataset de modelado:

  team_stats.parquet (N temporadas) ─┐
  stadiums.csv ──────────────────────┼─→ data/model/dataset.parquet
  weather.parquet ───────────────────┘

Una fila por (match_id, team_id, is_home). Target: throw_ins_total.

Features:
  - rolling{3,5,10}_{col}        — media de W partidos previos del equipo
  - ewma_alpha{03,05}_{col}      — EWMA con decay exponencial (pondera reciente)
  - std_{col}                    — season-to-date (media acumulada en temporada)
  - opp_*                        — las mismas features del rival
  - days_since_last_match, has_full_history, matchday_number
  - pitch_length_m, pitch_width_m, capacity (del estadio del local)
  - temperature_2m, wind_speed_10m, precipitation, relative_humidity_2m,
    weather_code (en hora de kickoff)

Columnas `col` base: throw_ins_total, corners_total, possession_pct,
passes_total, fouls_committed, aerials_total, touches_total.

Uso:
  python -m model.dataset_builder
  python -m model.dataset_builder --output data/model/dataset.parquet
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import pandas as pd

from model.features import (
    FEATURE_SOURCE_COLS,
    TARGET_COL,
    WEATHER_COLS,
    compute_context_features,
    compute_ewma,
    compute_opponent_features,
    compute_rolling,
    compute_season_to_date,
)

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
log = logging.getLogger("dataset_builder")

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "team_stats_glob": "data/whoscored_laliga/**/*_team_stats.parquet",
    "stadiums_path": "data/reference/stadiums.csv",
    "weather_path": "data/reference/weather.parquet",
    "output_path": "data/model/dataset.parquet",
}

EXPECTED_ROWS = 3622  # 1811 partidos × 2 filas (home + away)


# ─────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────

def load_team_stats() -> pd.DataFrame:
    files = sorted(glob.glob(CONFIG["team_stats_glob"], recursive=True))
    if not files:
        raise FileNotFoundError(f"no team_stats encontrados en {CONFIG['team_stats_glob']}")
    log.info("Cargando %d ficheros team_stats ...", len(files))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["match_date"] = pd.to_datetime(df["match_date"])
    log.info("team_stats cargado: %s", df.shape)
    return df


def load_stadiums() -> pd.DataFrame:
    s = pd.read_csv(CONFIG["stadiums_path"])
    if "whoscored_id" not in s.columns:
        raise ValueError("stadiums.csv sin columna whoscored_id")
    s = s.rename(columns={"whoscored_id": "stadium_team_id"})
    return s[["stadium_team_id", "pitch_length_m", "pitch_width_m", "capacity"]]


def load_weather() -> pd.DataFrame:
    w = pd.read_parquet(CONFIG["weather_path"])
    keep = ["match_id", *WEATHER_COLS]
    return w[[c for c in keep if c in w.columns]]


# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN
# ─────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    df = load_team_stats()
    stadiums = load_stadiums()
    weather = load_weather()

    if df[TARGET_COL].isna().any():
        raise ValueError(f"{TARGET_COL} tiene nulls en team_stats; el scraper debe repoblar")

    log.info("Calculando rolling features ...")
    df = compute_rolling(df)
    log.info("Calculando EWMA features ...")
    df = compute_ewma(df)
    log.info("Calculando season-to-date features ...")
    df = compute_season_to_date(df)
    log.info("Calculando context features ...")
    df = compute_context_features(df)
    log.info("Calculando opponent features ...")
    df = compute_opponent_features(df)

    # Matchday: del campo `round` del calendario si existe, si no cumcount por season
    df = _add_matchday_number(df)

    log.info("Join con estadios y weather ...")
    df["stadium_team_id"] = df.apply(
        lambda r: r["team_id"] if r["is_home"] == 1 else r["opponent_id"], axis=1
    )
    df = df.merge(stadiums, on="stadium_team_id", how="left")
    df = df.merge(weather, on="match_id", how="left")

    _validate(df)

    return df


def _add_matchday_number(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna `matchday_number` por equipo×temporada usando cumcount por orden cronológico."""
    df = df.sort_values(["season", "team_id", "match_date"]).reset_index(drop=True)
    df["matchday_number"] = df.groupby(["season", "team_id"]).cumcount() + 1
    return df


# ─────────────────────────────────────────────────────────────
# VALIDACIONES (asserts)
# ─────────────────────────────────────────────────────────────

def _validate(df: pd.DataFrame) -> None:
    log.info("Validando dataset ...")
    assert len(df) == EXPECTED_ROWS, f"filas esperadas {EXPECTED_ROWS}, recibidas {len(df)}"

    assert df[TARGET_COL].isna().sum() == 0, f"{TARGET_COL} tiene nulls"

    dup = df.duplicated(subset=["match_id", "is_home"]).sum()
    assert dup == 0, f"{dup} filas duplicadas por (match_id, is_home)"

    # Cada match_id debe aparecer exactamente 2 veces
    per_match = df.groupby("match_id").size()
    bad = per_match[per_match != 2]
    assert bad.empty, f"{len(bad)} match_ids con != 2 filas"

    # Stadium y weather sin nulls obligatorios
    stadium_cols = ["pitch_length_m", "pitch_width_m", "capacity"]
    for c in stadium_cols:
        nulls = df[c].isna().sum()
        assert nulls == 0, f"stadium {c} tiene {nulls} nulls"
    for c in WEATHER_COLS:
        if c in df.columns:
            nulls = df[c].isna().sum()
            assert nulls == 0, f"weather {c} tiene {nulls} nulls"

    log.info("Validación OK.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(output_path: str | None = None) -> None:
    out = Path(output_path or CONFIG["output_path"])
    df = build_dataset()

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    n_features = sum(1 for c in df.columns if c.startswith(("rolling", "ewma_", "std_", "opp_")))
    log.info("Dataset guardado: %s (shape %s)", out, df.shape)
    log.info("Columnas totales: %d | features rolling/ewma/std/opp: %d", len(df.columns), n_features)
    log.info("Target `%s` — mean %.2f std %.2f", TARGET_COL, df[TARGET_COL].mean(), df[TARGET_COL].std())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build modeling dataset")
    parser.add_argument("--output", default=CONFIG["output_path"], help="Ruta del parquet de salida")
    args = parser.parse_args()
    main(output_path=args.output)
