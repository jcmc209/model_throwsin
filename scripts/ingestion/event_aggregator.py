"""
Event Aggregator
================
Lee todos los parquets `all_events` de WhoScored y genera un parquet con
agregados por (match_id, team_id) que capturan el estilo de juego, las
acciones directamente relacionadas con los saques de banda, el tempo y el
game-state histórico.

Output: data/reference/event_stats.parquet
  match_id        int64
  team_id         int64
  crosses         int32  — nº de centros
  long_balls      int32  — nº de balones largos
  heads           int32  — nº de acciones de cabeza
  wide_events     int32  — nº de eventos en zonas Left+Right
  wide_ratio      float32 — wide_events / total_events
  avg_pass_length float32 — longitud media del pase (solo passes con length>0)
  avg_zone_x      float32 — posición media X del equipo en el campo (0=portería propia, 100=portería rival)
  std_y           float32 — dispersión lateral del equipo (std de Y, 0=banda izq, 100=banda der)
  total_events    int32  — nº total de eventos del equipo en el partido (proxy de tempo)
  leading_pct     float32 — fracción del partido en que el equipo iba ganando [0,1]

Justificación mecánica:
  - total_events → cuánto juega el equipo (tempo) → más acciones → más probable saque lateral
  - leading_pct → equipos que lideran juegan más posesional (r=-0.22 con throw_ins), los que
    van perdiendo presionan y generan más duelos en banda

Uso:
  python scripts/ingestion/event_aggregator.py
  python scripts/ingestion/event_aggregator.py --output data/reference/event_stats.parquet
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("event_aggregator")

CONFIG = {
    "events_glob": "data/whoscored_laliga/**/*_all_events.parquet",
    "throw_ins_glob": "data/whoscored_laliga/**/*_throw_ins.parquet",
    "output_path": "data/reference/event_stats.parquet",
}

EVENT_COLS = [
    "crosses",
    "long_balls",
    "heads",
    "wide_events",
    "wide_ratio",
    "avg_pass_length",
    "avg_zone_x",
    "std_y",
    # wide-atomic: conteos por tipo de evento en zona lateral (Left/Right)
    "aerial_wide",
    "dispossessed_wide",
    "takeon_wide",
    "balltouch_wide",
    "foul_wide",
    # tempo / game-state
    "total_events",
    "leading_pct",
    # B4: high-press recoveries (BallRecovery con x > 60)
    "press_recoveries_high",
    # B1: throw-in zone ratios
    "throw_in_def_zone_pct",
    "throw_in_att_zone_pct",
    # B5: throw-in retention (3 eventos)
    "throw_in_retention3_pct",
]


def load_all_events() -> pd.DataFrame:
    files = sorted(glob.glob(CONFIG["events_glob"], recursive=True))
    if not files:
        raise FileNotFoundError(f"No se encontraron all_events en {CONFIG['events_glob']}")
    log.info("Cargando %d archivos all_events ...", len(files))
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=[
            "match_id", "team_id", "event_id",
            "is_cross", "is_long_ball", "is_head",
            "zone", "pass_length", "x", "y",
            "event_type", "expanded_minute", "period",
            "is_home_team",
        ])
        dfs.append(df)
        log.info("  %s: %d eventos, %d partidos", f, len(df), df["match_id"].nunique())
    return pd.concat(dfs, ignore_index=True)


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    log.info("Agregando por (match_id, team_id) ...")

    events = events.copy()
    events["is_wide"] = events["zone"].isin(["Left", "Right"]).astype(int)
    events["pass_len_nonzero"] = events["pass_length"].where(events["pass_length"] > 0)

    # Flags para wide-atomic: 1 si el evento es de ese tipo EN zona lateral
    events["is_aerial_wide"] = ((events["event_type"] == "Aerial") & events["is_wide"]).astype(int)
    events["is_dispossessed_wide"] = ((events["event_type"] == "Dispossessed") & events["is_wide"]).astype(int)
    events["is_takeon_wide"] = ((events["event_type"] == "TakeOn") & events["is_wide"]).astype(int)
    events["is_balltouch_wide"] = ((events["event_type"] == "BallTouch") & events["is_wide"]).astype(int)
    events["is_foul_wide"] = ((events["event_type"] == "Foul") & events["is_wide"]).astype(int)

    # B4: high-press recoveries — BallRecovery en campo rival (x > 60)
    events["is_high_press_recovery"] = (
        (events["event_type"] == "BallRecovery") & (events["x"] > 60)
    ).astype(int)

    def std_y_safe(s: pd.Series) -> float:
        return float(s.std(ddof=0)) if len(s) > 1 else 0.0

    grp = events.groupby(["match_id", "team_id"])

    agg = pd.DataFrame({
        "crosses": grp["is_cross"].sum().astype("int32"),
        "long_balls": grp["is_long_ball"].sum().astype("int32"),
        "heads": grp["is_head"].sum().astype("int32"),
        "wide_events": grp["is_wide"].sum().astype("int32"),
        "total_events": grp["event_id"].count().astype("int32"),
        "avg_pass_length": grp["pass_len_nonzero"].mean().astype("float32"),
        "avg_zone_x": grp["x"].mean().astype("float32"),
        "std_y": grp["y"].apply(std_y_safe).astype("float32"),
        "aerial_wide": grp["is_aerial_wide"].sum().astype("int32"),
        "dispossessed_wide": grp["is_dispossessed_wide"].sum().astype("int32"),
        "takeon_wide": grp["is_takeon_wide"].sum().astype("int32"),
        "balltouch_wide": grp["is_balltouch_wide"].sum().astype("int32"),
        "foul_wide": grp["is_foul_wide"].sum().astype("int32"),
        "press_recoveries_high": grp["is_high_press_recovery"].sum().astype("int32"),
    }).reset_index()

    agg["wide_ratio"] = (agg["wide_events"] / agg["total_events"]).astype("float32")

    # Imputar nulls residuales
    pl_median = float(agg["avg_pass_length"].median())
    agg["avg_pass_length"] = agg["avg_pass_length"].fillna(pl_median)

    agg["total_events"] = agg["total_events"].astype("int32")
    # Excluye leading_pct (se añade después via compute_game_state_stats)
    # y las columnas de throw_ins (se añaden después via load_throw_ins)
    _throw_ins_cols = {"throw_in_def_zone_pct", "throw_in_att_zone_pct", "throw_in_retention3_pct"}
    base_cols = [c for c in EVENT_COLS if c not in {"leading_pct"} | _throw_ins_cols]
    return agg[["match_id", "team_id"] + base_cols]


def compute_game_state_stats(events: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula leading_pct por (match_id, team_id) usando el timeline de goles.

    leading_pct = minutos_liderando / duracion_partido ∈ [0, 1]

    Mecanismo: equipos que lideran tienden a jugar más posesional (menos saques).
    Equipos que van perdiendo presionan y generan más duelos en banda (más saques).

    Para partidos sin goles (0-0): leading_pct = 0.0 para ambos equipos (todo el
    partido en empate).
    """
    log.info("Calculando game-state stats (timeline de goles) ...")

    goals = events[events["event_type"] == "Goal"][
        ["match_id", "team_id", "is_home_team", "expanded_minute"]
    ].copy()

    # Duración efectiva del partido = max(expanded_minute) en 2ª parte
    duration = (
        events[events["period"] == "SecondHalf"]
        .groupby("match_id")["expanded_minute"]
        .max()
        .rename("match_duration")
    )

    # Equipos por partido (home/away) — necesario para imputar partidos sin goles
    teams_per_match = (
        events[["match_id", "team_id", "is_home_team"]]
        .drop_duplicates(subset=["match_id", "team_id"])
    )

    results: list[dict] = []

    for match_id, match_teams in teams_per_match.groupby("match_id"):
        dur = int(duration.get(match_id, 90))
        match_goals = goals[goals["match_id"] == match_id].sort_values("expanded_minute")

        # Mapa team_id → is_home para este partido
        home_team = match_teams[match_teams["is_home_team"] == 1]["team_id"].values
        away_team = match_teams[match_teams["is_home_team"] == 0]["team_id"].values

        if len(home_team) == 0 or len(away_team) == 0:
            # Sin información de home/away: leading_pct = 0 para todos
            for _, row in match_teams.iterrows():
                results.append({"match_id": match_id, "team_id": row["team_id"], "leading_pct": 0.0})
            continue

        home_id = int(home_team[0])
        away_id = int(away_team[0])

        home_leading = 0
        away_leading = 0
        home_score = 0
        away_score = 0
        prev_min = 0

        for _, goal_row in match_goals.iterrows():
            seg = int(goal_row["expanded_minute"]) - prev_min
            if seg > 0:
                if home_score > away_score:
                    home_leading += seg
                elif away_score > home_score:
                    away_leading += seg
            if int(goal_row["is_home_team"]) == 1:
                home_score += 1
            else:
                away_score += 1
            prev_min = int(goal_row["expanded_minute"])

        # Segmento final tras el último gol
        seg = dur - prev_min
        if seg > 0:
            if home_score > away_score:
                home_leading += seg
            elif away_score > home_score:
                away_leading += seg

        dur_f = max(dur, 1)
        results.append({"match_id": match_id, "team_id": home_id, "leading_pct": home_leading / dur_f})
        results.append({"match_id": match_id, "team_id": away_id, "leading_pct": away_leading / dur_f})

    gs_df = pd.DataFrame(results)
    gs_df["leading_pct"] = gs_df["leading_pct"].clip(0.0, 1.0).astype("float32")
    log.info("Game-state stats: %d filas, %d partidos", len(gs_df), gs_df["match_id"].nunique())
    return gs_df


def load_throw_ins() -> pd.DataFrame:
    """
    Agrega estadísticas de throw_ins por (match_id, team_id).

    Columnas generadas:
      - throw_in_def_zone_pct   — fracción de saques en tercio defensivo
      - throw_in_att_zone_pct   — fracción de saques en tercio atacante
      - throw_in_retention3_pct — media de retained_possession_3 (retención en 3 eventos)
    """
    files = sorted(glob.glob(CONFIG["throw_ins_glob"], recursive=True))
    if not files:
        log.warning("No se encontraron throw_ins parquets en %s", CONFIG["throw_ins_glob"])
        return pd.DataFrame()

    log.info("Cargando %d archivos throw_ins ...", len(files))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(
                f, columns=["match_id", "team_id", "field_zone", "retained_possession_3"]
            ))
        except Exception as exc:  # columnas faltantes en un archivo concreto
            log.warning("  Saltando %s: %s", f, exc)

    if not dfs:
        return pd.DataFrame()

    ti = pd.concat(dfs, ignore_index=True)
    log.info("throw_ins cargados: %d filas, %d partidos", len(ti), ti["match_id"].nunique())

    agg = (
        ti.groupby(["match_id", "team_id"])
        .agg(
            throw_in_def_zone_pct=("field_zone", lambda x: (x == "defensive_third").mean()),
            throw_in_att_zone_pct=("field_zone", lambda x: (x == "attacking_third").mean()),
            throw_in_retention3_pct=("retained_possession_3", "mean"),
        )
        .reset_index()
    )

    for col in ("throw_in_def_zone_pct", "throw_in_att_zone_pct", "throw_in_retention3_pct"):
        agg[col] = agg[col].astype("float32")

    return agg


def validate(agg: pd.DataFrame) -> None:
    log.info("Validando event_stats ...")
    assert "match_id" in agg.columns and "team_id" in agg.columns
    for c in EVENT_COLS:
        nulls = agg[c].isna().sum()
        assert nulls == 0, f"event_stats['{c}'] tiene {nulls} nulls"
    dup = agg.duplicated(subset=["match_id", "team_id"]).sum()
    assert dup == 0, f"{dup} filas duplicadas por (match_id, team_id)"
    log.info("Validación OK — %d filas, %d partidos", len(agg), agg["match_id"].nunique())


def main(output_path: str | None = None) -> None:
    events = load_all_events()
    agg = aggregate_events(events)
    gs = compute_game_state_stats(events)

    # Merge leading_pct into main aggregate
    agg = agg.merge(gs, on=["match_id", "team_id"], how="left")
    agg["leading_pct"] = agg["leading_pct"].fillna(0.0).astype("float32")

    # Merge throw_ins features (B1, B5)
    ti_agg = load_throw_ins()
    if not ti_agg.empty:
        agg = agg.merge(ti_agg, on=["match_id", "team_id"], how="left")
        for col in ("throw_in_def_zone_pct", "throw_in_att_zone_pct", "throw_in_retention3_pct"):
            agg[col] = agg[col].fillna(agg[col].median()).astype("float32")
    else:
        log.warning("throw_ins no disponibles — rellenando con 0.0")
        for col in ("throw_in_def_zone_pct", "throw_in_att_zone_pct", "throw_in_retention3_pct"):
            agg[col] = np.float32(0.0)

    agg = agg[["match_id", "team_id"] + EVENT_COLS]
    validate(agg)

    out = Path(output_path or CONFIG["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out, index=False)
    log.info("event_stats guardado en %s (shape %s)", out, agg.shape)
    log.info("Estadisticas medias: %s", {
        c: f"{agg[c].mean():.3f}" for c in EVENT_COLS
    })


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Aggregate WhoScored events per match/team")
    parser.add_argument("--output", default=CONFIG["output_path"])
    args = parser.parse_args()
    main(output_path=args.output)
