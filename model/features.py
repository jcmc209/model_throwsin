"""
Feature Engineering — Throw-In Predictor
========================================
Funciones puras reutilizadas por `dataset_builder.py` (offline) y `predict.py`
(online). Manteniendo todas las features aquí garantizamos que el entrenamiento
y la inferencia computen exactamente lo mismo.

Reglas críticas (anti-leakage):
  * Todo rolling / EWMA / season-to-date usa `.shift(1)` antes de calcular,
    de modo que la fila del partido M solo mira partidos < M.
  * Ninguna función lee columnas del partido actual como predictor.

Convenciones de entrada:
  El DataFrame `df` es formato largo: una fila por (match_id, team_id, is_home)
  ordenado ascendentemente por match_date dentro de cada team_id.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CONSTANTES COMPARTIDAS
# ─────────────────────────────────────────────────────────────

TARGET_COL = "throw_ins_total"

FEATURE_SOURCE_COLS: list[str] = [
    "throw_ins_total",
    "corners_total",
    "possession_pct",
    "passes_total",
    "fouls_committed",
    "aerials_total",
    "touches_total",
]

ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 10)
EWMA_ALPHAS: tuple[float, ...] = (0.3, 0.5)


# ─────────────────────────────────────────────────────────────
# SORT / ORDEN
# ─────────────────────────────────────────────────────────────

def _sorted_by_team_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena por team_id, match_date asc. Requisito para shift(1)."""
    required = {"team_id", "match_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"columnas requeridas ausentes: {missing}")
    return df.sort_values(["team_id", "match_date"], kind="mergesort").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# ROLLING (uniforme)
# ─────────────────────────────────────────────────────────────

def compute_rolling(
    df: pd.DataFrame,
    target_cols: Iterable[str] = FEATURE_SOURCE_COLS,
    windows: Iterable[int] = ROLLING_WINDOWS,
    group_col: str = "team_id",
) -> pd.DataFrame:
    """
    Añade columnas `rolling{W}_{col}` = media de los W partidos previos del
    mismo team_id. Usa `.shift(1)` para excluir la fila actual.
    """
    df = _sorted_by_team_date(df)
    grouped = df.groupby(group_col, sort=False)

    for col in target_cols:
        if col not in df.columns:
            log.warning("compute_rolling: columna '%s' ausente, se omite", col)
            continue
        shifted = grouped[col].shift(1)
        for w in windows:
            df[f"rolling{w}_{col}"] = (
                shifted.groupby(df[group_col], sort=False)
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df


# ─────────────────────────────────────────────────────────────
# EWMA (ponderación exponencial — pondera más lo reciente)
# ─────────────────────────────────────────────────────────────

def compute_ewma(
    df: pd.DataFrame,
    target_cols: Iterable[str] = FEATURE_SOURCE_COLS,
    alphas: Iterable[float] = EWMA_ALPHAS,
    group_col: str = "team_id",
) -> pd.DataFrame:
    """
    Añade columnas `ewma_alpha{AA}_{col}` (AA = alpha*10, ej: alpha03) con
    Exponentially Weighted Moving Average. Partidos recientes pesan más.
    Usa shift(1) para anti-leakage.
    """
    df = _sorted_by_team_date(df)
    grouped = df.groupby(group_col, sort=False)

    for col in target_cols:
        if col not in df.columns:
            log.warning("compute_ewma: columna '%s' ausente, se omite", col)
            continue
        shifted = grouped[col].shift(1)
        for alpha in alphas:
            tag = f"alpha{int(alpha * 10):02d}"
            df[f"ewma_{tag}_{col}"] = (
                shifted.groupby(df[group_col], sort=False)
                .apply(lambda s: s.ewm(alpha=alpha, adjust=False, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
    return df


# ─────────────────────────────────────────────────────────────
# SEASON-TO-DATE
# ─────────────────────────────────────────────────────────────

def compute_season_to_date(
    df: pd.DataFrame,
    target_cols: Iterable[str] = FEATURE_SOURCE_COLS,
) -> pd.DataFrame:
    """
    Añade `std_{col}` = media acumulada del equipo en la temporada actual
    hasta el partido anterior. Reinicia en cada temporada. Shift(1).
    """
    if "season" not in df.columns:
        raise ValueError("compute_season_to_date requiere columna 'season'")
    df = _sorted_by_team_date(df)
    grouped = df.groupby(["team_id", "season"], sort=False)

    for col in target_cols:
        if col not in df.columns:
            continue
        shifted = grouped[col].shift(1)
        df[f"std_{col}"] = (
            shifted.groupby([df["team_id"], df["season"]], sort=False)
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    return df


# ─────────────────────────────────────────────────────────────
# FEATURES DEL OPONENTE
# ─────────────────────────────────────────────────────────────

def compute_opponent_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fila (match_id, team_id, is_home) añade las mismas rolling/EWMA/
    season-to-date del RIVAL (mismo match_id, is_home invertido) con prefijo
    `opp_`.

    Requiere que `compute_rolling`, `compute_ewma` y `compute_season_to_date`
    ya se hayan ejecutado.
    """
    required = {"match_id", "is_home"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"columnas requeridas ausentes: {missing}")

    feature_prefixes = ("rolling", "ewma_", "std_")
    feature_cols = [
        c for c in df.columns if c.startswith(feature_prefixes) and not c.startswith("opp_")
    ]
    if not feature_cols:
        log.warning("compute_opponent_features: no se encontraron features base; no-op")
        return df

    left = df[["match_id", "is_home"] + feature_cols].copy()
    left["is_home_opp"] = 1 - left["is_home"]
    opp = left.drop(columns=["is_home"]).rename(
        columns={"is_home_opp": "is_home", **{c: f"opp_{c}" for c in feature_cols}}
    )

    merged = df.merge(opp, on=["match_id", "is_home"], how="left")
    return merged


# ─────────────────────────────────────────────────────────────
# CONTEXTO (días descanso, matchday, has_full_history)
# ─────────────────────────────────────────────────────────────

def compute_context_features(df: pd.DataFrame, min_history_matches: int = 5) -> pd.DataFrame:
    """
    Añade:
      - days_since_last_match: delta en días frente al partido anterior del equipo
      - has_full_history: True si team_id tiene >= min_history_matches partidos previos
                          en el dataset
    """
    df = _sorted_by_team_date(df).copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    prev_date = df.groupby("team_id", sort=False)["match_date"].shift(1)
    df["days_since_last_match"] = (df["match_date"] - prev_date).dt.days.fillna(-1).astype(int)

    match_rank = df.groupby("team_id", sort=False).cumcount()
    df["has_full_history"] = match_rank >= min_history_matches
    return df


# ─────────────────────────────────────────────────────────────
# IMPUTACIÓN WEATHER PARA PARTIDOS >16 DÍAS (sin forecast)
# ─────────────────────────────────────────────────────────────

WEATHER_COLS = (
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "relative_humidity_2m",
    "weather_code",
)


def impute_weather_forecast_gap(
    df: pd.DataFrame,
    historical_weather: pd.DataFrame,
    stadium_col: str = "stadium_team_id",
) -> pd.DataFrame:
    """
    Rellena columnas meteorológicas faltantes con la media histórica de ese
    estadio en el mes del partido. Añade flag `weather_imputed`.

    Args:
      df: filas de inferencia con match_date + stadium_col + columnas weather
          (posiblemente NaN para partidos >16d).
      historical_weather: DataFrame con temperature_2m, wind_speed_10m, etc.,
                          match_date y stadium_col. Fuente de la media.

    Returns:
      df con WEATHER_COLS rellenas y columna `weather_imputed` (bool).
    """
    df = df.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    historical_weather = historical_weather.copy()
    historical_weather["match_date"] = pd.to_datetime(historical_weather["match_date"])
    historical_weather["month"] = historical_weather["match_date"].dt.month

    agg_cols = [c for c in WEATHER_COLS if c in historical_weather.columns]
    monthly = (
        historical_weather.groupby([stadium_col, "month"])[agg_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"_imp_{c}" for c in agg_cols})
    )

    df["month"] = df["match_date"].dt.month
    before_null = df[list(agg_cols)].isna().any(axis=1)
    df = df.merge(monthly, on=[stadium_col, "month"], how="left")

    for col in agg_cols:
        df[col] = df[col].fillna(df[f"_imp_{col}"])
    df["weather_imputed"] = before_null & df[list(agg_cols)].notna().all(axis=1)

    df = df.drop(columns=[c for c in df.columns if c.startswith("_imp_")] + ["month"])
    return df


# ─────────────────────────────────────────────────────────────
# UTILIDAD: lista de columnas feature generadas
# ─────────────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Devuelve las columnas que son features del modelo (no target, no IDs)."""
    drop_cols = {
        TARGET_COL,
        "match_id",
        "team_id",
        "opponent_id",
        "team_name",
        "opponent_name",
        "match_date",
        "season",
        "result_score",
        "ft_score",
        "ht_score",
        "venue",
        "stadium_team_id",
    }
    return [c for c in df.columns if c not in drop_cols and not np.issubdtype(df[c].dtype, np.dtype("O"))]
