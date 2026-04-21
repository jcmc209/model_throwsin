"""
Dataset Builder Match-Level — Throw-In Predictor (Bivariate)
============================================================
Genera `data/model/dataset_match.parquet` con 1 fila por partido reutilizando
`data/model/dataset.parquet` (formato largo, 2 filas por partido).

Estructura de salida:
  - Columnas de identificación: match_id, season, match_date, home_team_id,
    away_team_id, home_team_name, away_team_name.
  - Features match-level (idénticas en ambas filas del dataset original, se
    toman de la fila home): matchday_number, referee_id, ref_*, capacity,
    pitch_length_m, pitch_width_m, temperature_2m, wind_speed_10m,
    precipitation, relative_humidity_2m, weather_code.
  - Features per-team duplicadas con prefijo `home_`/`away_`: rolling{3,5,10}_*,
    ewma_*, std_*, days_since_last_match, has_full_history.
  - Diagnóstico: home_throw_ins_total, away_throw_ins_total.
  - Target principal: throw_ins_total_match = home_ti + away_ti.
  - Target secundario (share): share_home = home_ti / (home_ti + away_ti).

Uso:
  python -m model.dataset_builder_match
  python -m model.dataset_builder_match --output data/model/dataset_match.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("dataset_builder_match")


CONFIG = {
    "input_path": "data/model/dataset.parquet",
    "output_path": "data/model/dataset_match.parquet",
}

EXPECTED_ROWS = 1811  # 3622 / 2

# Features que son match-level (idénticas en home/away en dataset.parquet)
MATCH_LEVEL_COLS = [
    "matchday_number",
    "referee_id",
    "ref_rolling5_throw_ins",
    "ref_rolling10_throw_ins",
    "ref_ewma_throw_ins",
    "ref_matches_count",
    "pitch_length_m",
    "pitch_width_m",
    "capacity",
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "relative_humidity_2m",
    "weather_code",
]

# Prefijos de features per-team (se duplican home_ / away_)
PER_TEAM_PREFIXES = ("rolling", "ewma_", "std_")

# Columnas adicionales per-team a duplicar (fuera de los prefijos anteriores)
EXTRA_PER_TEAM_COLS = ["days_since_last_match", "has_full_history"]

# Columnas que NO se pivotan ni se incluyen en el match dataset
EXCLUDE_COLS = {
    # IDs / meta que se gestionan aparte
    "match_id", "season", "match_date", "team_id", "team_name",
    "opponent_id", "opponent_name", "is_home",
    # Resultado / marcadores
    "result_score", "ft_score", "ht_score", "venue",
    # Estadio base (el stadium es del equipo local, se usa tal cual)
    "stadium_team_id",
    # Target base (lo manejamos aparte como home_/away_throw_ins_total)
    "throw_ins_total",
}


def _collect_per_team_cols(df: pd.DataFrame) -> list[str]:
    """Columnas per-team (rolling/ewma/std + opp_* + extras) a duplicar home/away."""
    per_team = [
        c for c in df.columns
        if (c.startswith(PER_TEAM_PREFIXES) or c.startswith("opp_"))
        and c not in EXCLUDE_COLS
    ]
    for c in EXTRA_PER_TEAM_COLS:
        if c in df.columns and c not in per_team:
            per_team.append(c)
    return per_team


def build_match_dataset() -> pd.DataFrame:
    df = pd.read_parquet(CONFIG["input_path"])
    log.info("Dataset largo cargado: %s", df.shape)

    per_team_cols = _collect_per_team_cols(df)
    log.info("Features per-team a duplicar home/away: %d", len(per_team_cols))

    # Split home/away
    home = df[df["is_home"] == 1].copy()
    away = df[df["is_home"] == 0].copy()
    log.info("Filas home=%d, away=%d", len(home), len(away))

    # Sanity: cada match_id debe aparecer en ambas mitades
    assert set(home["match_id"]) == set(away["match_id"]), \
        "match_ids no coinciden entre home y away"

    # Base con identificación + match-level features (tomadas de home)
    base_cols = ["match_id", "season", "match_date"] + MATCH_LEVEL_COLS
    base = home[base_cols + ["team_id", "team_name"]].rename(
        columns={"team_id": "home_team_id", "team_name": "home_team_name"}
    )

    # away_team_id / away_team_name
    away_ids = away[["match_id", "team_id", "team_name"]].rename(
        columns={"team_id": "away_team_id", "team_name": "away_team_name"}
    )

    # Per-team features home/away
    home_feats = home[["match_id"] + per_team_cols + ["throw_ins_total"]].rename(
        columns={
            **{c: f"home_{c}" for c in per_team_cols},
            "throw_ins_total": "home_throw_ins_total",
        }
    )
    away_feats = away[["match_id"] + per_team_cols + ["throw_ins_total"]].rename(
        columns={
            **{c: f"away_{c}" for c in per_team_cols},
            "throw_ins_total": "away_throw_ins_total",
        }
    )

    match_df = (
        base
        .merge(away_ids, on="match_id", how="inner")
        .merge(home_feats, on="match_id", how="inner")
        .merge(away_feats, on="match_id", how="inner")
    )

    # Targets
    match_df["throw_ins_total_match"] = (
        match_df["home_throw_ins_total"].astype(int)
        + match_df["away_throw_ins_total"].astype(int)
    )
    match_df["share_home"] = (
        match_df["home_throw_ins_total"]
        / match_df["throw_ins_total_match"].replace(0, pd.NA)
    ).astype(float)

    # Orden estable por fecha
    match_df = match_df.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    _validate(match_df, per_team_cols)
    return match_df


def _validate(df: pd.DataFrame, per_team_cols: list[str]) -> None:
    log.info("Validando match dataset ...")
    assert len(df) == EXPECTED_ROWS, \
        f"filas esperadas {EXPECTED_ROWS}, recibidas {len(df)}"
    assert df["match_id"].nunique() == len(df), "match_id duplicados en match dataset"

    # Target principal sin nulls
    assert df["throw_ins_total_match"].isna().sum() == 0, \
        "throw_ins_total_match tiene nulls"

    # Match-level features sin nulls (heredan validación de dataset.parquet)
    for c in ("capacity", "pitch_length_m", "pitch_width_m",
              "temperature_2m", "weather_code", "matchday_number",
              "ref_rolling5_throw_ins", "ref_ewma_throw_ins"):
        if c in df.columns:
            nulls = df[c].isna().sum()
            assert nulls == 0, f"match-level {c} tiene {nulls} nulls"

    # Sanity: home_X + away_X no todo ceros para rolling principal
    if "home_rolling5_throw_ins_total" in df.columns:
        suma = (df["home_rolling5_throw_ins_total"].fillna(0)
                + df["away_rolling5_throw_ins_total"].fillna(0))
        nonzero_ratio = (suma > 0).mean()
        assert nonzero_ratio > 0.8, \
            f"sanity: <80% de matches con rolling5 ti_total > 0 ({nonzero_ratio:.2%})"

    log.info("Validación OK: %d partidos, %d columnas", len(df), len(df.columns))
    log.info(
        "Target throw_ins_total_match — mean %.2f | std %.2f | min %d | max %d",
        df["throw_ins_total_match"].mean(),
        df["throw_ins_total_match"].std(),
        int(df["throw_ins_total_match"].min()),
        int(df["throw_ins_total_match"].max()),
    )
    log.info(
        "share_home — mean %.4f | std %.4f",
        df["share_home"].mean(),
        df["share_home"].std(),
    )


def main(output_path: str | None = None) -> None:
    out = Path(output_path or CONFIG["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_match_dataset()
    df.to_parquet(out, index=False)
    log.info("Match dataset guardado: %s (shape %s)", out, df.shape)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Build match-level dataset (bivariate)")
    parser.add_argument("--output", default=CONFIG["output_path"])
    args = parser.parse_args()
    main(output_path=args.output)
