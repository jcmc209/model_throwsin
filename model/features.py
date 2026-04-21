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

# Columnas agregadas desde all_events con vínculo mecánico directo a saques de banda.
# Se añaden al dataset via event_aggregator.py → data/reference/event_stats.parquet.
# Se aplican las mismas funciones rolling/EWMA/std_ que a FEATURE_SOURCE_COLS.
EVENT_FEATURE_SOURCE_COLS: list[str] = [
    "crosses",           # nº de centros (balón por banda → saque si sale)
    "long_balls",        # nº de balones largos (duelo aéreo → más probable saque)
    "heads",             # nº de acciones de cabeza (duelo aéreo, superset de aerials)
    "wide_events",       # nº de eventos en zonas Left+Right (mide juego por banda)
    "wide_ratio",        # wide_events / total_events (normalizado por volumen)
    "avg_pass_length",   # longitud media de pase (estilo directo vs posesión)
    "avg_zone_x",        # posición media X del equipo (presión alta → más saques en campo rival)
    "std_y",             # dispersión lateral (equipo que usa las bandas vs central)
    # wide-atomic: conteos por tipo de evento en zona lateral (r > 0.23 con target)
    "aerial_wide",       # duelos aéreos en banda (r=+0.26)
    "dispossessed_wide", # pérdidas contestadas en banda (r=+0.26)
    "takeon_wide",       # regates en banda (r=+0.24)
    "balltouch_wide",    # toques en zona lateral (r=+0.33)
    "foul_wide",         # faltas en banda (r=+0.16)
]

ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 10)
EWMA_ALPHAS: tuple[float, ...] = (0.3, 0.5)
REF_EWMA_ALPHA: float = 0.3

# Top-30 features seleccionadas por análisis SHAP (mean |SHAP| sobre 400 muestras de train).
# Orden: descendente por importancia. Cubren: throw-ins rolling largo, aerials (propio+oponente),
# contexto (is_home, matchday, capacity), meteorología, árbitro y posesión del rival.
# Actualizar ejecutando: python scripts/shap_analysis.py (si se añaden nuevas temporadas).
SHAP_SELECTED_FEATURES: list[str] = [
    # event-based (derivadas de all_events, vínculo mecánico directo a saques)
    "std_y",                          # dispersión lateral — máxima importancia SHAP
    "opp_std_y",                      # dispersión lateral del rival
    "std_heads",                      # cabezazos season-to-date
    "std_avg_pass_length",            # longitud media de pase season-to-date
    "std_long_balls",                 # balones largos season-to-date
    "opp_rolling10_crosses",          # centros del rival (rolling 10)
    "rolling5_wide_ratio",            # % acciones por banda (rolling 5)
    "opp_std_long_balls",             # balones largos del rival season-to-date
    "opp_rolling5_avg_pass_length",   # longitud de pase del rival
    "opp_ewma_alpha03_heads",         # cabezazos del rival (EWMA)
    "ewma_alpha05_crosses",           # centros propios (EWMA)
    "opp_ewma_alpha05_avg_zone_x",    # posición media X del rival
    "opp_rolling3_wide_ratio",        # % acciones por banda del rival (rolling 3)
    "std_wide_ratio",                 # % acciones por banda season-to-date
    "opp_ewma_alpha05_heads",         # cabezazos del rival (EWMA α=0.5)
    # wide-atomic (Bloque A — toques/regates/duelos en zona lateral)
    "opp_rolling10_balltouch_wide",   # toques en banda del rival (rolling 10) — Δ -0.0041
    "ewma_alpha03_balltouch_wide",    # toques en banda propios (EWMA α=0.3)    — Δ -0.0009
    # autorregresivo (throw-ins históricos)
    "rolling5_throw_ins_total",
    "std_throw_ins_total",
    "opp_ewma_alpha03_throw_ins_total",
    "opp_rolling5_throw_ins_total",
    "rolling10_throw_ins_total",
    # aerials (proxy indirecto — superviviente del SHAP anterior)
    "opp_std_aerials_total",
    "opp_rolling10_aerials_total",
    "opp_rolling3_aerials_total",
    "std_aerials_total",
    "opp_ewma_alpha03_aerials_total",
    # contextuales / estadio / meteorología
    "is_home",
    "capacity",
    "wind_speed_10m",
    "days_since_last_match",
    # posesión / pase
    "opp_ewma_alpha05_possession_pct",
    "opp_ewma_alpha05_passes_total",
    "rolling5_passes_total",
    "std_fouls_committed",
    "opp_ewma_alpha05_fouls_committed",
    # árbitro
    "ref_rolling5_throw_ins",
]


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
# FEATURES DE ÁRBITRO (rolling histórico del árbitro)
# ─────────────────────────────────────────────────────────────

REF_FEATURE_COLS = (
    "ref_rolling5_throw_ins",
    "ref_rolling10_throw_ins",
    "ref_ewma_throw_ins",
    "ref_matches_count",
)


def compute_referee_features(df: pd.DataFrame, referee_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Añade 4 features históricas del árbitro a `df`:
      - ref_rolling5_throw_ins   media de las 5 últimas actuaciones del árbitro
      - ref_rolling10_throw_ins  media de las 10 últimas
      - ref_ewma_throw_ins       EWMA α=0.3 del árbitro
      - ref_matches_count        nº de partidos previos dirigidos por este árbitro

    Los valores son match-level: idénticos en las filas home y away del mismo match_id.
    Anti-leakage: shift(1) dentro del grupo referee_id antes del rolling.

    Args:
        df:             dataset largo (una fila por match_id × is_home), debe incluir match_id.
        referee_stats:  DataFrame con match_id, referee_id, match_date, throw_ins_total_match.

    Returns:
        df con las 4 columnas ref_* añadidas.
    """
    required = {"match_id", "referee_id", "match_date", "throw_ins_total_match"}
    missing = required - set(referee_stats.columns)
    if missing:
        raise ValueError(f"referee_stats faltan columnas: {missing}")

    # Construir historial del árbitro ordenado cronológicamente
    ref = (
        referee_stats[["match_id", "referee_id", "match_date", "throw_ins_total_match"]]
        .dropna(subset=["referee_id"])
        .sort_values("match_date", kind="mergesort")
        .reset_index(drop=True)
    )
    ref["referee_id"] = ref["referee_id"].astype(int)

    grp = ref.groupby("referee_id", sort=False)
    shifted = grp["throw_ins_total_match"].shift(1)

    ref["ref_rolling5_throw_ins"] = (
        shifted.groupby(ref["referee_id"], sort=False)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    ref["ref_rolling10_throw_ins"] = (
        shifted.groupby(ref["referee_id"], sort=False)
        .rolling(window=10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    ref["ref_ewma_throw_ins"] = (
        shifted.groupby(ref["referee_id"], sort=False)
        .apply(lambda s: s.ewm(alpha=REF_EWMA_ALPHA, adjust=False, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    ref["ref_matches_count"] = (
        grp.cumcount()
    )

    # Para la primera actuación de cada árbitro (shift → NaN), usar media global como prior
    global_mean = float(ref["throw_ins_total_match"].mean())
    for col in ("ref_rolling5_throw_ins", "ref_rolling10_throw_ins", "ref_ewma_throw_ins"):
        ref[col] = ref[col].fillna(global_mean)

    ref_features = ref[
        ["match_id", "referee_id", "ref_rolling5_throw_ins",
         "ref_rolling10_throw_ins", "ref_ewma_throw_ins", "ref_matches_count"]
    ]

    # Join al dataset (match-level → mismo valor home y away)
    df = df.merge(ref_features, on="match_id", how="left", suffixes=("", "_ref_dup"))

    # Columna referee_id puede llegar sin duplicado si no estaba ya en df
    if "referee_id_ref_dup" in df.columns:
        df = df.drop(columns=["referee_id_ref_dup"])

    # Imputar con media global para matches sin árbitro registrado en referee_stats
    for col in ("ref_rolling5_throw_ins", "ref_rolling10_throw_ins", "ref_ewma_throw_ins"):
        df[col] = df[col].fillna(global_mean)
    df["ref_matches_count"] = df["ref_matches_count"].fillna(0).astype(int)

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
# STYLE FEATURES (índice de juego directo)
# ─────────────────────────────────────────────────────────────

def compute_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade 2 features de estilo de juego:
      - rolling5_direct_play      rolling5 de aerials/(passes+1) del equipo
      - opp_rolling5_direct_play  ídem del oponente en el mismo partido

    Ambas usan shift(1) para anti-leakage. El índice aerials/passes captura
    cuánto "juega directo" un equipo en relación a su volumen de pases —
    ortogonal a los aerials absolutos ya presentes en SHAP_SELECTED_FEATURES.

    Requiere columnas: team_id, match_id, is_home, match_date,
                       aerials_total, passes_total.
    """
    required = {"team_id", "match_id", "is_home", "match_date",
                "aerials_total", "passes_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_style_features: columnas ausentes {missing}")

    df = _sorted_by_team_date(df).copy()

    # Índice bruto por fila (post-partido — solo para calcular rolling histórico)
    df["_direct_play_raw"] = df["aerials_total"] / (df["passes_total"] + 1.0)

    global_mean = float(df["_direct_play_raw"].mean())

    shifted = df.groupby("team_id", sort=False)["_direct_play_raw"].shift(1)
    df["rolling5_direct_play"] = (
        shifted.groupby(df["team_id"], sort=False)
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rolling5_direct_play"] = df["rolling5_direct_play"].fillna(global_mean)

    df = df.drop(columns=["_direct_play_raw"])

    # Versión del oponente: merge invertido por is_home
    own = df[["match_id", "is_home", "rolling5_direct_play"]].copy()
    own["is_home_opp"] = 1 - own["is_home"]
    opp = own.drop(columns=["is_home"]).rename(
        columns={"is_home_opp": "is_home",
                 "rolling5_direct_play": "opp_rolling5_direct_play"}
    )
    df = df.merge(opp, on=["match_id", "is_home"], how="left")
    df["opp_rolling5_direct_play"] = df["opp_rolling5_direct_play"].fillna(global_mean)

    return df


# ─────────────────────────────────────────────────────────────
# H2H: historial directo entre el mismo par de rivales
# ─────────────────────────────────────────────────────────────

H2H_FEATURE_COLS = (
    "h2h_avg_throw_ins",   # media expanding de saques en previos enfrentamientos directos
    "h2h_last3_throw_ins", # media de los últimos 3 enfrentamientos directos
    "h2h_count",           # nº de enfrentamientos directos previos (regularizador)
)


def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada fila (match_id, team_id) añade el historial de saques en enfrentamientos
    previos entre el MISMO par de equipos (team_id vs opponent_id).

    Features añadidas:
      - h2h_avg_throw_ins   — expanding mean de throw_ins_total en partidos anteriores
                              entre este par exacto (en el rol team_id, no oponente).
      - h2h_last3_throw_ins — rolling(3) mean de los últimos 3 enfrentamientos.
      - h2h_count           — nº de encuentros directos previos disponibles.

    Anti-leakage: shift(1) dentro del grupo (team_id, opponent_id) antes de cualquier
    agregación, de modo que la fila del partido M solo mira partidos < M.

    Imputación: cuando no hay historial H2H (primeros enfrentamientos o equipos
    ascendidos), se imputa con std_throw_ins_total del propio equipo. Si std_throw_ins_total
    tampoco está disponible, se usa la media global del dataset.

    Requiere columnas: team_id, opponent_id, match_date, throw_ins_total.
    """
    required = {"team_id", "opponent_id", "match_date", TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_h2h_features: columnas ausentes {missing}")

    df = _sorted_by_team_date(df).copy()

    global_mean = float(df[TARGET_COL].mean())

    # Agrupar por par ordenado (team_id, opponent_id) y ordenar cronológicamente
    grp = df.groupby(["team_id", "opponent_id"], sort=False)

    shifted = grp[TARGET_COL].shift(1)

    df["h2h_avg_throw_ins"] = (
        shifted.groupby([df["team_id"], df["opponent_id"]], sort=False)
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["h2h_last3_throw_ins"] = (
        shifted.groupby([df["team_id"], df["opponent_id"]], sort=False)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["h2h_count"] = (
        grp[TARGET_COL].cumcount()
    )

    # Imputar NaN (primeros enfrentamientos) con std_throw_ins_total del equipo si existe
    if "std_throw_ins_total" in df.columns:
        fallback = df["std_throw_ins_total"].fillna(global_mean)
    else:
        fallback = global_mean

    df["h2h_avg_throw_ins"] = df["h2h_avg_throw_ins"].fillna(fallback)
    df["h2h_last3_throw_ins"] = df["h2h_last3_throw_ins"].fillna(fallback)

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
