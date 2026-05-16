"""
Weather Fetcher — Open-Meteo
============================
Descarga condiciones meteorológicas en la hora de kickoff para cada
partido de LaLiga (2021/22–2025/26) usando la API gratuita de Open-Meteo.

Estrategia: batching por estadio × temporada.
  ~100 llamadas totales (20 estadios × 5 temporadas) en vez de 1.811 individuales.

  - Partidos pasados (> 7 días)  → archive-api.open-meteo.com  (ERA5 reanalysis)
  - Partidos recientes / futuros → api.open-meteo.com          (forecast, 16 días)
  - Partidos > 16 días futuro    → se dejan null; relanzar el fetcher más adelante

Kickoff hour:
  - Temporadas 2023/24–2025/26: extraído de liga_calendar_rows.csv
  - Temporadas 2021/22–2022/23: fijo a las 20:00 (sin hora en el calendario)

Output: data/reference/weather.parquet
  Una fila por match_id con columnas:
    match_id, temperature_2m, wind_speed_10m, precipitation,
    relative_humidity_2m, weather_code, kickoff_hour, source

Uso:
  python weather_fetcher.py            # descarga todo
  python weather_fetcher.py --dry-run  # muestra cuántas llamadas haría sin descargar
"""
from __future__ import annotations

import glob
import logging
import time
import argparse
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("weather_fetcher.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "stadiums_path":    "data/reference/stadiums.csv",
    "calendar_path":    "data/reference/liga_calendar_rows.csv",
    "team_stats_glob":  "data/whoscored_laliga/**/*_team_stats.parquet",
    "output_path":      "data/reference/weather.parquet",

    "weather_vars": [
        "temperature_2m",
        "wind_speed_10m",
        "precipitation",
        "relative_humidity_2m",
        "weather_code",
    ],

    # Temporadas sin hora de kickoff en el calendario (2021/22, 2022/23)
    "default_kickoff_hour": 20,

    # El archivo ERA5 tiene un retraso de ~7 días; usar forecast para lo más reciente
    "archive_delay_days": 7,

    # Open-Meteo forecast cubre hasta 16 días adelante
    "forecast_horizon_days": 16,

    # Pausa entre llamadas API (cortesía)
    "api_delay_s": 0.5,
}

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────

def load_stadiums() -> pd.DataFrame:
    """
    Carga stadiums.csv.
    Devuelve solo filas con whoscored_id y coordenadas completas.
    calendar_name fallback: si está vacío usa club.
    """
    df = pd.read_csv(CONFIG["stadiums_path"])
    df = df[df["whoscored_id"].notna() & df["latitude"].notna() & df["longitude"].notna()].copy()
    df["whoscored_id"] = df["whoscored_id"].astype(int)
    df["calendar_name"] = df["calendar_name"].fillna(df["club"])
    return df


def load_calendar() -> pd.DataFrame:
    """
    Carga liga_calendar_rows.csv filtrado a LaLiga (sin postponed).
    Extrae kickoff_hour como entero.
    """
    df = pd.read_csv(CONFIG["calendar_path"])
    df = df[
        (df["competition"] == "La Liga") &
        (df["status"] != "postponed")
    ].copy()
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    df["kickoff_hour"] = pd.to_datetime(df["match_time"], format="%H:%M:%S").dt.hour
    return df[["home_team", "match_date", "kickoff_hour"]].copy()


def load_all_matches() -> pd.DataFrame:
    """
    Carga team_stats de todas las temporadas.
    Devuelve un DataFrame con una fila por partido (equipo local únicamente):
      match_id, match_date (date), home_team_id (int), season (str)
    """
    frames = []
    for f in glob.glob(CONFIG["team_stats_glob"], recursive=True):
        ts = pd.read_parquet(
            f, columns=["match_id", "match_date", "team_id", "is_home", "season"]
        )
        frames.append(ts)

    all_ts = pd.concat(frames, ignore_index=True)
    home = all_ts[all_ts["is_home"] == 1][
        ["match_id", "match_date", "team_id", "season"]
    ].drop_duplicates("match_id")

    home["match_date"] = pd.to_datetime(home["match_date"].str[:10]).dt.date
    home["team_id"] = home["team_id"].astype(int)
    return home.rename(columns={"team_id": "home_team_id"})


def load_existing_weather() -> set[int]:
    """Devuelve los match_ids que ya están en weather.parquet."""
    path = Path(CONFIG["output_path"])
    if not path.exists():
        return set()
    existing = pd.read_parquet(path, columns=["match_id"])
    return set(existing["match_id"].tolist())

# ─────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL SCHEDULE
# ─────────────────────────────────────────────────────────────

def build_schedule(
    matches: pd.DataFrame,
    calendar: pd.DataFrame,
    stadiums: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combina matches + calendar + stadiums para obtener:
      match_id, match_date, home_team_id, season,
      latitude, longitude, stadium_name, kickoff_hour
    """
    # ── 1. Añadir coordenadas de estadio ──────────────────────
    sta = stadiums[["whoscored_id", "stadium_name", "latitude", "longitude", "calendar_name"]]
    schedule = matches.merge(
        sta, left_on="home_team_id", right_on="whoscored_id", how="left"
    )

    # ── 2. Resolver kickoff_hour desde el calendario ──────────
    # Mapeamos calendar.home_team → whoscored_id via calendar_name
    cal_with_id = calendar.merge(
        stadiums[["calendar_name", "whoscored_id"]],
        left_on="home_team",
        right_on="calendar_name",
        how="left",
    ).rename(columns={"whoscored_id": "home_team_id"})

    # Índice de búsqueda: (match_date, home_team_id) → kickoff_hour
    cal_idx = (
        cal_with_id.dropna(subset=["home_team_id"])
        .set_index(["match_date", "home_team_id"])["kickoff_hour"]
    )

    def _resolve_kickoff(row) -> int:
        key = (row["match_date"], row["home_team_id"])
        if key in cal_idx.index:
            return int(cal_idx[key])
        return CONFIG["default_kickoff_hour"]

    schedule["kickoff_hour"] = schedule.apply(_resolve_kickoff, axis=1)

    return schedule

# ─────────────────────────────────────────────────────────────
# LLAMADAS A LA API
# ─────────────────────────────────────────────────────────────

def _call_api(
    lat: float,
    lon: float,
    date_from: date,
    date_to: date,
    use_forecast: bool,
) -> dict | None:
    """Llama a Open-Meteo y devuelve el JSON completo o None si falla."""
    url = FORECAST_URL if use_forecast else ARCHIVE_URL
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join(CONFIG["weather_vars"]),
        "timezone":   "Europe/Madrid",
        "start_date": str(date_from),
        "end_date":   str(date_to),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning("  API error [%s → %s]: %s", date_from, date_to, exc)
        return None


def _extract_hour(data: dict, target_date: date, target_hour: int) -> dict | None:
    """
    Extrae los valores de una hora concreta del JSON de Open-Meteo.
    Devuelve un dict con las weather_vars o None si no se encuentra.
    """
    if not data or "hourly" not in data:
        return None
    hourly = data["hourly"]
    target_ts = f"{target_date}T{target_hour:02d}:00"
    try:
        idx = hourly["time"].index(target_ts)
    except ValueError:
        return None
    return {var: hourly[var][idx] for var in CONFIG["weather_vars"]}

# ─────────────────────────────────────────────────────────────
# FETCH POR BATCH (estadio × temporada)
# ─────────────────────────────────────────────────────────────

def fetch_batch(
    lat: float,
    lon: float,
    group: pd.DataFrame,
) -> list[dict]:
    """
    Dado un grupo de partidos del mismo estadio y temporada:
    - Hace una llamada al archivo para los partidos pasados (> archive_delay_days)
    - Hace una llamada al forecast para los partidos recientes/próximos (≤ 16 días)
    - Salta los partidos más allá del horizonte de forecast
    Devuelve lista de dicts con match_id + variables meteorológicas.
    """
    today           = date.today()
    archive_cutoff  = today - timedelta(days=CONFIG["archive_delay_days"])
    forecast_cutoff = today + timedelta(days=CONFIG["forecast_horizon_days"])

    past   = group[group["match_date"] <= archive_cutoff]
    recent = group[
        (group["match_date"] > archive_cutoff) &
        (group["match_date"] <= forecast_cutoff)
    ]
    far_future = group[group["match_date"] > forecast_cutoff]

    if not far_future.empty:
        log.info(
            "  %d partidos más allá del horizonte de forecast — sin datos por ahora",
            len(far_future),
        )

    results: list[dict] = []

    # ── Histórico ─────────────────────────────────────────────
    if not past.empty:
        raw = _call_api(lat, lon, past["match_date"].min(), past["match_date"].max(), use_forecast=False)
        time.sleep(CONFIG["api_delay_s"])
        for _, row in past.iterrows():
            parsed = _extract_hour(raw, row["match_date"], row["kickoff_hour"])
            if parsed:
                results.append({"match_id": row["match_id"], "kickoff_hour": row["kickoff_hour"], "source": "historical", **parsed})
            else:
                log.warning("  Sin datos históricos para match_id=%s (%s h%s)", row["match_id"], row["match_date"], row["kickoff_hour"])

    # ── Reciente / Forecast ───────────────────────────────────
    if not recent.empty:
        raw = _call_api(lat, lon, recent["match_date"].min(), recent["match_date"].max(), use_forecast=True)
        time.sleep(CONFIG["api_delay_s"])
        for _, row in recent.iterrows():
            parsed = _extract_hour(raw, row["match_date"], row["kickoff_hour"])
            if parsed:
                results.append({"match_id": row["match_id"], "kickoff_hour": row["kickoff_hour"], "source": "forecast", **parsed})
            else:
                log.warning("  Sin forecast para match_id=%s (%s h%s)", row["match_id"], row["match_date"], row["kickoff_hour"])

    return results

# ─────────────────────────────────────────────────────────────
# GUARDAR / MERGE
# ─────────────────────────────────────────────────────────────

def save_weather(new_rows: list[dict], output_path: Path) -> None:
    """
    Append incremental: combina nuevas filas con el parquet existente.
    Idempotente: si match_id ya existe, mantiene la fila existente.
    """
    if not new_rows:
        log.info("Nada nuevo que guardar.")
        return

    new_df = pd.DataFrame(new_rows)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        combined = (
            pd.concat([existing, new_df], ignore_index=True)
            .drop_duplicates(subset=["match_id"], keep="first")
        )
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    log.info("weather.parquet guardado: %d filas totales.", len(combined))

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    log.info("=== Weather Fetcher%s ===", " (dry-run)" if dry_run else "")

    stadiums  = load_stadiums()
    calendar  = load_calendar()
    matches   = load_all_matches()
    done_ids  = load_existing_weather()

    log.info("Estadios con coordenadas:  %d", len(stadiums))
    log.info("Entradas en calendario:    %d", len(calendar))
    log.info("Partidos únicos (home):    %d", len(matches))
    log.info("Ya con weather:            %d", len(done_ids))

    schedule = build_schedule(matches, calendar, stadiums)

    missing_coords = schedule[schedule["latitude"].isna()]
    if not missing_coords.empty:
        log.warning(
            "%d partidos sin coordenadas de estadio: %s",
            len(missing_coords),
            missing_coords.groupby("season").size().to_dict(),
        )

    pending = schedule[~schedule["match_id"].isin(done_ids)]
    log.info("Partidos pendientes:       %d", len(pending))

    if dry_run:
        groups = pending.dropna(subset=["latitude"]).groupby(["stadium_name", "season"]).size()
        log.info("Dry-run: %d llamadas API estimadas (2 por batch con datos mixtos).", len(groups))
        print("\n" + groups.to_string())
        return

    if pending.empty:
        log.info("Nada que hacer.")
        return

    all_rows: list[dict] = []

    batches = list(pending.dropna(subset=["latitude"]).groupby(["stadium_name", "season"]))
    for (stadium, season), group in tqdm(batches, desc="Fetching weather"):
        lat = float(group["latitude"].iloc[0])
        lon = float(group["longitude"].iloc[0])
        log.info("Procesando: %s — %s (%d partidos)", stadium, season, len(group))
        rows = fetch_batch(lat, lon, group)
        all_rows.extend(rows)

    save_weather(all_rows, Path(CONFIG["output_path"]))
    log.info("Completado. %d partidos nuevos con meteorología.", len(all_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga meteorología en la hora de kickoff para partidos de LaLiga."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo muestra cuántas llamadas API haría, sin descargar datos.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
