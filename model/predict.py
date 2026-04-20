"""
Predict — Throw-In Predictor
============================
Inferencia para partidos futuros (status='scheduled' en el calendario).

Pipeline:
  1. Carga modelo entrenado (data/model/model_v1.joblib)
  2. Carga calendario, filtra LaLiga scheduled (por --date / --matchday)
  3. Resuelve team_id de cada equipo usando stadiums.calendar_name → whoscored_id
  4. Para cada (partido, equipo), construye features rolling/EWMA usando
     SOLO partidos con fecha < fecha del partido a predecir (sin leakage)
  5. Une weather disponible; imputa media estadio-mes si falta (>16d vista)
  6. Valida esquema contra model["features"], predice, guarda

Uso:
  python -m model.predict --matchday next
  python -m model.predict --date 2026-05-01
  python -m model.predict --all-scheduled
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.dataset_builder import load_stadiums, load_team_stats, load_weather
from model.features import (
    FEATURE_SOURCE_COLS,
    TARGET_COL,
    WEATHER_COLS,
    compute_context_features,
    compute_ewma,
    compute_opponent_features,
    compute_rolling,
    compute_season_to_date,
    impute_weather_forecast_gap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("predict")

CONFIG = {
    "model_path": "data/model/model_v1.joblib",
    "calendar_path": "data/reference/liga_calendar_rows.csv",
    "stadiums_path": "data/reference/stadiums.csv",
    "weather_path": "data/reference/weather.parquet",
    "output_dir": "data/model",
}


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

def load_model(path: str | None = None) -> dict:
    p = Path(path or CONFIG["model_path"])
    if not p.exists():
        log.error("Modelo no encontrado en %s — ejecuta `python -m model.train` primero.", p)
        sys.exit(2)
    artifact = joblib.load(p)
    log.info("Modelo cargado: %s (val MAE %.4f, entrenado %s)",
             artifact.get("version", "?"), artifact.get("val_mae", float("nan")),
             artifact.get("trained_at", "?"))
    return artifact


def load_scheduled_matches(
    date_filter: str | None,
    matchday_next: bool,
    all_scheduled: bool,
) -> pd.DataFrame:
    cal = pd.read_csv(CONFIG["calendar_path"])
    cal["match_date"] = pd.to_datetime(cal["match_date"])
    cal = cal[
        (cal["status"] == "scheduled")
        & (cal["competition"].str.contains("La Liga", case=False, na=False))
    ].copy()

    if date_filter:
        target = pd.to_datetime(date_filter).normalize()
        cal = cal[cal["match_date"] == target]
    elif matchday_next:
        # Próxima jornada: los scheduled con la fecha mínima y los 9 partidos siguientes dentro de una ventana de 7 días
        if cal.empty:
            return cal
        earliest = cal["match_date"].min()
        cal = cal[cal["match_date"] <= earliest + pd.Timedelta(days=7)]
    elif not all_scheduled:
        raise ValueError("Especifica --date, --matchday next o --all-scheduled")

    log.info("Partidos scheduled a predecir: %d", len(cal))
    return cal.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# RESOLUCIÓN DE team_id DESDE CALENDAR NAMES
# ─────────────────────────────────────────────────────────────

def build_team_name_map() -> dict[str, int]:
    """Mapa calendar_name → whoscored_id usando stadiums.csv."""
    s = pd.read_csv(CONFIG["stadiums_path"])
    if "calendar_name" not in s.columns or "whoscored_id" not in s.columns:
        raise ValueError("stadiums.csv debe tener columnas 'calendar_name' y 'whoscored_id'")
    return dict(zip(s["calendar_name"], s["whoscored_id"]))


# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DE FILAS DE INFERENCIA
# ─────────────────────────────────────────────────────────────

def build_inference_rows(scheduled: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte scheduled (1 fila por partido) en formato largo (2 filas por partido,
    home+away) que se pueda concatenar con `history` y pasar por las funciones de
    features.

    Los partidos scheduled tendrán NaN en las columnas de estadísticas base; eso es
    correcto porque las features se calculan usando SOLO partidos anteriores (shift(1)
    en rolling/EWMA), y la fila del partido actual nunca se usa como fuente.
    """
    name_to_id = build_team_name_map()

    missing_home = set(scheduled["home_team"]) - set(name_to_id)
    missing_away = set(scheduled["away_team"]) - set(name_to_id)
    missing = missing_home | missing_away
    if missing:
        log.warning("Equipos sin match en stadiums.csv (se ignoran): %s", missing)
        scheduled = scheduled[
            scheduled["home_team"].isin(name_to_id) & scheduled["away_team"].isin(name_to_id)
        ].copy()

    scheduled["home_team_id"] = scheduled["home_team"].map(name_to_id).astype(int)
    scheduled["away_team_id"] = scheduled["away_team"].map(name_to_id).astype(int)
    scheduled["match_id"] = scheduled["event_id"]

    rows = []
    for _, r in scheduled.iterrows():
        for is_home, team_id, opp_id in (
            (1, r["home_team_id"], r["away_team_id"]),
            (0, r["away_team_id"], r["home_team_id"]),
        ):
            rows.append({
                "match_id": int(r["match_id"]),
                "season": r["season"],
                "match_date": pd.to_datetime(r["match_date"]),
                "team_id": int(team_id),
                "team_name": r["home_team"] if is_home else r["away_team"],
                "opponent_id": int(opp_id),
                "opponent_name": r["away_team"] if is_home else r["home_team"],
                "is_home": is_home,
            })
    inf_df = pd.DataFrame(rows)

    # Concatenar con histórico para que rolling/EWMA tengan contexto
    common_cols = [c for c in history.columns if c in inf_df.columns]
    missing_in_inf = [c for c in history.columns if c not in inf_df.columns]
    for c in missing_in_inf:
        inf_df[c] = np.nan
    inf_df = inf_df[history.columns]

    combined = pd.concat([history, inf_df], ignore_index=True)
    combined = combined.sort_values(["team_id", "match_date"]).reset_index(drop=True)
    return combined, set(zip(inf_df["match_id"], inf_df["is_home"]))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(
    date_filter: str | None = None,
    matchday_next: bool = False,
    all_scheduled: bool = False,
    output_path: str | None = None,
) -> None:
    artifact = load_model()
    model = artifact["model"]
    expected_features = artifact["features"]

    scheduled = load_scheduled_matches(date_filter, matchday_next, all_scheduled)
    if scheduled.empty:
        log.warning("Sin partidos scheduled a predecir.")
        return

    log.info("Cargando histórico team_stats para construir rolling ...")
    history = load_team_stats()

    combined, inference_ids = build_inference_rows(scheduled, history)

    log.info("Calculando features rolling/EWMA/std/context/opponent ...")
    combined = compute_rolling(combined)
    combined = compute_ewma(combined)
    combined = compute_season_to_date(combined)
    combined = compute_context_features(combined)
    combined = compute_opponent_features(combined)

    # Filtrar a solo filas de inferencia
    combined["_row_key"] = list(zip(combined["match_id"], combined["is_home"]))
    pred_df = combined[combined["_row_key"].isin(inference_ids)].copy()
    pred_df = pred_df.drop(columns=["_row_key"])

    # Join con stadiums + weather
    stadiums = load_stadiums()
    pred_df["stadium_team_id"] = pred_df.apply(
        lambda r: r["team_id"] if r["is_home"] == 1 else r["opponent_id"], axis=1
    )
    pred_df = pred_df.merge(stadiums, on="stadium_team_id", how="left")

    weather = load_weather()
    pred_df = pred_df.merge(weather, on="match_id", how="left")

    # Imputar weather para partidos >16 días (sin forecast)
    historical_with_stadium = history.copy()
    historical_with_stadium = historical_with_stadium.merge(
        weather, on="match_id", how="left"
    )
    historical_with_stadium["stadium_team_id"] = historical_with_stadium.apply(
        lambda r: r["team_id"] if r["is_home"] == 1 else r["opponent_id"], axis=1
    )
    pred_df = impute_weather_forecast_gap(pred_df, historical_with_stadium)

    # Matchday number aproximado por cumcount en temporada (igual que en training)
    pred_df = pred_df.sort_values(["season", "team_id", "match_date"])
    pred_df["matchday_number"] = (
        pred_df.groupby(["season", "team_id"]).cumcount() + 1
    )

    # Validar que todas las features esperadas existen
    missing = [c for c in expected_features if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Features ausentes tras construcción: {missing[:5]} ... ({len(missing)} total)")
    X = pred_df[expected_features].astype(float)

    preds = model.predict(X)
    pred_df["prediction"] = preds

    # Reconstruir formato ancho: pred_home / pred_away / total
    wide = pred_df.pivot_table(
        index=["match_id", "match_date", "season"],
        columns="is_home",
        values="prediction",
        aggfunc="first",
    ).rename(columns={0: "pred_throw_ins_away", 1: "pred_throw_ins_home"}).reset_index()

    names = pred_df.pivot_table(
        index=["match_id"], columns="is_home", values="team_name", aggfunc="first"
    ).rename(columns={0: "away_team", 1: "home_team"}).reset_index()

    out = wide.merge(names, on="match_id")
    out["pred_throw_ins_total"] = out["pred_throw_ins_home"] + out["pred_throw_ins_away"]
    out = out[[
        "match_id", "match_date", "season", "home_team", "away_team",
        "pred_throw_ins_home", "pred_throw_ins_away", "pred_throw_ins_total",
    ]].sort_values("match_date").reset_index(drop=True)

    out_dir = Path(output_path or CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"predictions_{stamp}.parquet"
    out.to_parquet(out_path, index=False)

    log.info("Predicciones guardadas en %s (%d partidos)", out_path, len(out))
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Predict throw-ins for scheduled matches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Fecha específica YYYY-MM-DD")
    group.add_argument("--matchday", choices=["next"], help="Próxima jornada")
    group.add_argument("--all-scheduled", action="store_true", help="Todos los scheduled")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    args = parser.parse_args()

    main(
        date_filter=args.date,
        matchday_next=args.matchday == "next",
        all_scheduled=args.all_scheduled,
        output_path=args.output_dir,
    )
