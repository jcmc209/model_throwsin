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
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.dataset_builder import load_event_stats, load_referees, load_stadiums, load_team_stats, load_weather
from model.features import (
    EVENT_FEATURE_SOURCE_COLS,
    FEATURE_SOURCE_COLS,
    TARGET_COL,
    WEATHER_COLS,
    compute_context_features,
    compute_ewma,
    compute_h2h_features,
    compute_opponent_features,
    compute_referee_features,
    compute_rolling,
    compute_season_to_date,
    compute_style_features,
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
    "model_total_path": "data/model/model_v1_total.joblib",
    "share_coefs_path": "data/model/share_coefs.json",
    "model_q25_path": "data/model/model_q25.joblib",
    "model_q50_path": "data/model/model_q50.joblib",
    "model_q75_path": "data/model/model_q75.joblib",
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

    # Normalizar referee_name: columna vacía → None
    if "referee_name" not in cal.columns:
        cal["referee_name"] = None
    else:
        cal["referee_name"] = cal["referee_name"].replace("", None).where(
            cal["referee_name"].notna() & (cal["referee_name"].str.strip() != ""), None
        )

    n_with_ref = cal["referee_name"].notna().sum()
    log.info(
        "Partidos scheduled a predecir: %d (%d con árbitro designado, %d sin árbitro)",
        len(cal), n_with_ref, len(cal) - n_with_ref,
    )
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
        ref_name = r.get("referee_name", None)
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
                "_referee_name": ref_name,  # propagado para inyectar en referee_features
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
# REFEREE INJECTION — asigna referee_id a partidos futuros cuando
# el árbitro se conoce de antemano (columna referee_name en el calendario)
# ─────────────────────────────────────────────────────────────

def inject_known_referee(
    combined: pd.DataFrame,
    referee_stats: pd.DataFrame,
    inference_ids: set,
) -> pd.DataFrame:
    """
    Para filas de inferencia con `_referee_name` relleno, busca el referee_id
    en referee_stats (match por nombre exacto, case-insensitive) y lo asigna.
    Esto permite que compute_referee_features calcule el historial real del árbitro
    en lugar de imputar con la media global.
    """
    if "_referee_name" not in combined.columns:
        return combined

    # Tabla nombre → referee_id (único por nombre en referee_stats)
    name_to_id = (
        referee_stats.dropna(subset=["referee_name"])
        .drop_duplicates(subset=["referee_name"])
        .set_index("referee_name")["referee_id"]
        .to_dict()
    )
    # Versión case-insensitive
    name_to_id_lower = {k.lower(): v for k, v in name_to_id.items()}

    combined = combined.copy()
    if "referee_id" not in combined.columns:
        combined["referee_id"] = np.nan

    mask_inference = combined.apply(
        lambda r: (r["match_id"], r["is_home"]) in inference_ids, axis=1
    )
    for idx, row in combined[mask_inference & combined["_referee_name"].notna()].iterrows():
        name = str(row["_referee_name"]).strip()
        rid = name_to_id.get(name) or name_to_id_lower.get(name.lower())
        if rid is not None:
            combined.at[idx, "referee_id"] = rid
            log.info("Árbitro inyectado: '%s' → referee_id=%s (match_id=%s)", name, rid, int(row["match_id"]))
        else:
            log.warning("Árbitro '%s' no encontrado en referee_stats — se usará media global", name)

    return combined


# ─────────────────────────────────────────────────────────────
# BIVARIATE INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────

def _pivot_long_to_match(long_df: pd.DataFrame) -> pd.DataFrame:
    """Convierte `pred_df` (formato largo, 2 filas por match) a match-level
    con prefijos home_/away_ reutilizando la misma lógica que `dataset_builder_match`.
    """
    from model.dataset_builder_match import (
        EXTRA_PER_TEAM_COLS, MATCH_LEVEL_COLS, PER_TEAM_PREFIXES,
    )

    per_team_cols = [
        c for c in long_df.columns
        if (c.startswith(PER_TEAM_PREFIXES) or c.startswith("opp_"))
    ]
    for c in EXTRA_PER_TEAM_COLS:
        if c in long_df.columns and c not in per_team_cols:
            per_team_cols.append(c)

    home = long_df[long_df["is_home"] == 1].copy()
    away = long_df[long_df["is_home"] == 0].copy()

    match_level = [c for c in MATCH_LEVEL_COLS if c in home.columns]
    base = home[["match_id", "season", "match_date", "team_id", "team_name"] + match_level].rename(
        columns={"team_id": "home_team_id", "team_name": "home_team_name"}
    )
    away_ids = away[["match_id", "team_id", "team_name"]].rename(
        columns={"team_id": "away_team_id", "team_name": "away_team_name"}
    )
    home_feats = home[["match_id"] + per_team_cols].rename(
        columns={c: f"home_{c}" for c in per_team_cols}
    )
    away_feats = away[["match_id"] + per_team_cols].rename(
        columns={c: f"away_{c}" for c in per_team_cols}
    )

    return (
        base.merge(away_ids, on="match_id", how="inner")
            .merge(home_feats, on="match_id", how="inner")
            .merge(away_feats, on="match_id", how="inner")
    )


def _apply_share_coefs(match_df: pd.DataFrame, coefs: dict) -> np.ndarray:
    """Aplica share_coefs.json al match_df y devuelve pred_share clipado a [0, 1]."""
    features = coefs["features"]
    X = pd.DataFrame(index=match_df.index)
    # Réplica exacta de _compute_share_features del training
    if "possession_diff" in features:
        X["possession_diff"] = (
            match_df["home_rolling5_possession_pct"].fillna(
                match_df["home_rolling5_possession_pct"].median()
            )
            - match_df["away_rolling5_possession_pct"].fillna(
                match_df["away_rolling5_possession_pct"].median()
            )
        )
    if "home_rolling_diff" in features:
        X["home_rolling_diff"] = (
            match_df["home_rolling5_throw_ins_total"].fillna(
                match_df["home_rolling5_throw_ins_total"].median()
            )
            - match_df["away_rolling5_throw_ins_total"].fillna(
                match_df["away_rolling5_throw_ins_total"].median()
            )
        )

    pred = np.full(len(match_df), float(coefs["intercept"]))
    for f in features:
        pred = pred + X[f].to_numpy() * float(coefs["coefs"][f])
    lo, hi = coefs.get("clip_range", [0.0, 1.0])
    return np.clip(pred, lo, hi)


# ─────────────────────────────────────────────────────────────
# MAIN (per-team)
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

    log.info("Mergeando event_stats histórico ...")
    event_stats = load_event_stats()
    history = history.merge(event_stats, on=["match_id", "team_id"], how="left")
    for c in EVENT_FEATURE_SOURCE_COLS:
        if c in history.columns:
            history[c] = history[c].fillna(history[c].median())

    combined, inference_ids = build_inference_rows(scheduled, history)

    all_source_cols = FEATURE_SOURCE_COLS + EVENT_FEATURE_SOURCE_COLS

    log.info("Calculando features rolling/EWMA/std/context/opponent/referee ...")
    combined = compute_rolling(combined, target_cols=all_source_cols)
    combined = compute_ewma(combined, target_cols=all_source_cols)
    combined = compute_season_to_date(combined, target_cols=all_source_cols)
    combined = compute_h2h_features(combined)
    combined = compute_context_features(combined)
    combined = compute_opponent_features(combined)
    combined = compute_style_features(combined)

    referee_stats = load_referees()
    combined = inject_known_referee(combined, referee_stats, inference_ids)
    combined = compute_referee_features(combined, referee_stats)

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


# ─────────────────────────────────────────────────────────────
# MAIN (bivariate)
# ─────────────────────────────────────────────────────────────

def main_bivariate(
    date_filter: str | None = None,
    matchday_next: bool = False,
    all_scheduled: bool = False,
    output_path: str | None = None,
) -> None:
    # Cargar Model1 (total) y share coefs
    tp = Path(CONFIG["model_total_path"])
    sp = Path(CONFIG["share_coefs_path"])
    if not tp.exists() or not sp.exists():
        log.error(
            "Artifacts bivariate no encontrados (%s, %s). "
            "Ejecuta `python -m model.train --target bivariate` primero.", tp, sp,
        )
        sys.exit(2)

    total_artifact = joblib.load(tp)
    total_model = total_artifact["model"]
    expected_features = total_artifact["features"]
    with open(sp, "r", encoding="utf-8") as f:
        share_coefs = json.load(f)
    log.info(
        "Bivariate cargado — Model1 total MAE val %.4f | share intercept %.4f",
        total_artifact.get("val_total_mae", float("nan")),
        share_coefs["intercept"],
    )

    scheduled = load_scheduled_matches(date_filter, matchday_next, all_scheduled)
    if scheduled.empty:
        log.warning("Sin partidos scheduled a predecir.")
        return

    log.info("Cargando histórico team_stats ...")
    history = load_team_stats()

    log.info("Mergeando event_stats histórico ...")
    event_stats = load_event_stats()
    history = history.merge(event_stats, on=["match_id", "team_id"], how="left")
    for c in EVENT_FEATURE_SOURCE_COLS:
        if c in history.columns:
            history[c] = history[c].fillna(history[c].median())

    combined, inference_ids = build_inference_rows(scheduled, history)

    all_source_cols = FEATURE_SOURCE_COLS + EVENT_FEATURE_SOURCE_COLS

    log.info("Calculando features rolling/EWMA/std/context/opponent/referee ...")
    combined = compute_rolling(combined, target_cols=all_source_cols)
    combined = compute_ewma(combined, target_cols=all_source_cols)
    combined = compute_season_to_date(combined, target_cols=all_source_cols)
    combined = compute_h2h_features(combined)
    combined = compute_context_features(combined)
    combined = compute_opponent_features(combined)
    combined = compute_style_features(combined)

    referee_stats = load_referees()
    combined = inject_known_referee(combined, referee_stats, inference_ids)
    combined = compute_referee_features(combined, referee_stats)

    combined["_row_key"] = list(zip(combined["match_id"], combined["is_home"]))
    pred_df = combined[combined["_row_key"].isin(inference_ids)].copy()
    pred_df = pred_df.drop(columns=["_row_key"])

    stadiums = load_stadiums()
    pred_df["stadium_team_id"] = pred_df.apply(
        lambda r: r["team_id"] if r["is_home"] == 1 else r["opponent_id"], axis=1
    )
    pred_df = pred_df.merge(stadiums, on="stadium_team_id", how="left")

    weather = load_weather()
    pred_df = pred_df.merge(weather, on="match_id", how="left")

    historical_with_stadium = history.copy()
    historical_with_stadium = historical_with_stadium.merge(
        weather, on="match_id", how="left"
    )
    historical_with_stadium["stadium_team_id"] = historical_with_stadium.apply(
        lambda r: r["team_id"] if r["is_home"] == 1 else r["opponent_id"], axis=1
    )
    pred_df = impute_weather_forecast_gap(pred_df, historical_with_stadium)

    pred_df = pred_df.sort_values(["season", "team_id", "match_date"])
    pred_df["matchday_number"] = (
        pred_df.groupby(["season", "team_id"]).cumcount() + 1
    )

    # Pivotar a match-level (schema = dataset_match.parquet)
    match_df = _pivot_long_to_match(pred_df)

    missing = [c for c in expected_features if c not in match_df.columns]
    if missing:
        raise ValueError(
            f"Features ausentes en match_df de inferencia: {missing[:5]} ({len(missing)} total)"
        )
    X_total = match_df[expected_features].astype(float)

    pred_total = total_model.predict(X_total)
    pred_share = _apply_share_coefs(match_df, share_coefs)

    pred_home = pred_total * pred_share
    pred_away = pred_total * (1.0 - pred_share)

    out = pd.DataFrame({
        "match_id": match_df["match_id"].to_numpy(),
        "match_date": pd.to_datetime(match_df["match_date"]).to_numpy(),
        "season": match_df["season"].to_numpy(),
        "home_team": match_df["home_team_name"].to_numpy(),
        "away_team": match_df["away_team_name"].to_numpy(),
        "pred_throw_ins_home": pred_home,
        "pred_throw_ins_away": pred_away,
        "pred_throw_ins_total": pred_total,
    }).sort_values("match_date").reset_index(drop=True)

    out_dir = Path(output_path or CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"predictions_bivariate_{stamp}.parquet"
    out.to_parquet(out_path, index=False)

    log.info("Predicciones bivariate guardadas en %s (%d partidos)", out_path, len(out))
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


# ─────────────────────────────────────────────────────────────
# MAIN (quantile — intervalos de confianza Q25/Q50/Q75)
# ─────────────────────────────────────────────────────────────

def _load_quantile_models() -> dict[str, dict]:
    """Carga los 3 modelos cuantil. Falla con mensaje claro si no existen."""
    paths = {
        "q25": CONFIG["model_q25_path"],
        "q50": CONFIG["model_q50_path"],
        "q75": CONFIG["model_q75_path"],
    }
    models = {}
    for label, path in paths.items():
        p = Path(path)
        if not p.exists():
            log.error(
                "Modelo %s no encontrado en %s — ejecuta "
                "`python -m model.train_quantile` primero.", label, path,
            )
            sys.exit(2)
        models[label] = joblib.load(p)
    return models


def main_quantile(
    date_filter: str | None = None,
    matchday_next: bool = False,
    all_scheduled: bool = False,
    output_path: str | None = None,
) -> None:
    """
    Genera predicciones con intervalos de confianza (Q25/Q50/Q75) para los
    partidos programados. Útil para estrategias over/under:
      - total_Q25 > línea → señal OVER (75% confianza histórica)
      - total_Q75 < línea → señal UNDER (75% confianza histórica)
      - Entre Q25 y Q75 → no apostar (incertidumbre alta)
    """
    quantile_models = _load_quantile_models()
    expected_features = quantile_models["q50"]["features"]

    scheduled = load_scheduled_matches(date_filter, matchday_next, all_scheduled)
    if scheduled.empty:
        log.warning("Sin partidos scheduled a predecir.")
        return

    log.info("Cargando histórico team_stats para construir rolling ...")
    history = load_team_stats()

    log.info("Mergeando event_stats histórico ...")
    event_stats = load_event_stats()
    history = history.merge(event_stats, on=["match_id", "team_id"], how="left")
    for c in EVENT_FEATURE_SOURCE_COLS:
        if c in history.columns:
            history[c] = history[c].fillna(history[c].median())

    combined, inference_ids = build_inference_rows(scheduled, history)

    all_source_cols = FEATURE_SOURCE_COLS + EVENT_FEATURE_SOURCE_COLS

    log.info("Calculando features ...")
    combined = compute_rolling(combined, target_cols=all_source_cols)
    combined = compute_ewma(combined, target_cols=all_source_cols)
    combined = compute_season_to_date(combined, target_cols=all_source_cols)
    combined = compute_h2h_features(combined)
    combined = compute_context_features(combined)
    combined = compute_opponent_features(combined)
    combined = compute_style_features(combined)

    referee_stats = load_referees()
    combined = inject_known_referee(combined, referee_stats, inference_ids)
    combined = compute_referee_features(combined, referee_stats)

    combined["_row_key"] = list(zip(combined["match_id"], combined["is_home"]))
    pred_df = combined[combined["_row_key"].isin(inference_ids)].copy()
    pred_df = pred_df.drop(columns=["_row_key"])

    stadiums = load_stadiums()
    pred_df["stadium_team_id"] = pred_df.apply(
        lambda r: r["team_id"] if r["is_home"] == 1 else r["opponent_id"], axis=1
    )
    pred_df = pred_df.merge(stadiums, on="stadium_team_id", how="left")

    weather = load_weather()
    pred_df = pred_df.merge(weather, on="match_id", how="left")

    historical_with_stadium = history.copy()
    historical_with_stadium = historical_with_stadium.merge(weather, on="match_id", how="left")
    historical_with_stadium["stadium_team_id"] = historical_with_stadium.apply(
        lambda r: r["team_id"] if r["is_home"] == 1 else r["opponent_id"], axis=1
    )
    pred_df = impute_weather_forecast_gap(pred_df, historical_with_stadium)

    pred_df = pred_df.sort_values(["season", "team_id", "match_date"])
    pred_df["matchday_number"] = (
        pred_df.groupby(["season", "team_id"]).cumcount() + 1
    )

    missing = [c for c in expected_features if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Features ausentes: {missing[:5]} ({len(missing)} total)")
    X = pred_df[expected_features].astype(float)

    # Predicciones por cuantil
    pred_df["pred_q25"] = quantile_models["q25"]["model"].predict(X)
    pred_df["pred_q50"] = quantile_models["q50"]["model"].predict(X)
    pred_df["pred_q75"] = quantile_models["q75"]["model"].predict(X)

    # Pivotar a formato ancho (home / away)
    def _pivot_col(col: str, new_names: dict[int, str]) -> pd.DataFrame:
        return pred_df.pivot_table(
            index="match_id", columns="is_home", values=col, aggfunc="first"
        ).rename(columns=new_names).reset_index()

    q25_wide = _pivot_col("pred_q25", {0: "away_q25", 1: "home_q25"})
    q50_wide = _pivot_col("pred_q50", {0: "away_q50", 1: "home_q50"})
    q75_wide = _pivot_col("pred_q75", {0: "away_q75", 1: "home_q75"})

    meta = pred_df.pivot_table(
        index="match_id", columns="is_home", values="team_name", aggfunc="first"
    ).rename(columns={0: "away_team", 1: "home_team"}).reset_index()

    dates = pred_df[["match_id", "match_date", "season"]].drop_duplicates("match_id")

    out = (
        dates
        .merge(meta, on="match_id")
        .merge(q25_wide, on="match_id")
        .merge(q50_wide, on="match_id")
        .merge(q75_wide, on="match_id")
    )

    # Totales a nivel partido + clip anti-cruce
    out["total_q25"] = out["home_q25"] + out["away_q25"]
    out["total_q50"] = out["home_q50"] + out["away_q50"]
    out["total_q75"] = out["home_q75"] + out["away_q75"]
    out["total_q25"] = np.minimum(out["total_q25"], out["total_q50"])
    out["total_q75"] = np.maximum(out["total_q75"], out["total_q50"])

    out = out.sort_values("match_date").reset_index(drop=True)

    out_dir = Path(output_path or CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    out_path = out_dir / f"predictions_quantile_{stamp}.parquet"
    out.to_parquet(out_path, index=False)

    log.info("Predicciones cuantil guardadas en %s (%d partidos)", out_path, len(out))
    display_cols = [
        "match_date", "home_team", "away_team",
        "total_q25", "total_q50", "total_q75",
    ]
    print(out[display_cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print("\nInterpretación: total_Q25 > línea → OVER | total_Q75 < línea → UNDER | entre Q25-Q75 → no apostar")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Predict throw-ins for scheduled matches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Fecha específica YYYY-MM-DD")
    group.add_argument("--matchday", choices=["next"], help="Próxima jornada")
    group.add_argument("--all-scheduled", action="store_true", help="Todos los scheduled")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument(
        "--bivariate", action="store_true",
        help="Usa el modelo bivariate (Model1 total + share factor)",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "quantile"],
        default="standard",
        help="standard: predicción puntual (default); quantile: intervalos Q25/Q50/Q75",
    )
    args = parser.parse_args()

    if args.mode == "quantile":
        main_quantile(
            date_filter=args.date,
            matchday_next=args.matchday == "next" if args.matchday else False,
            all_scheduled=args.all_scheduled,
            output_path=args.output_dir,
        )
    elif args.bivariate:
        main_bivariate(
            date_filter=args.date,
            matchday_next=args.matchday == "next" if args.matchday else False,
            all_scheduled=args.all_scheduled,
            output_path=args.output_dir,
        )
    else:
        main(
            date_filter=args.date,
            matchday_next=args.matchday == "next" if args.matchday else False,
            all_scheduled=args.all_scheduled,
            output_path=args.output_dir,
        )
