"""
Odds Scheduler — auto-fetch de cuotas antes de cada partido
===========================================================
Bucle ligero que:
  1. Lee `data/reference/liga_calendar_rows.csv`.
  2. Filtra partidos scheduled de LaLiga en un horizonte acorde a las ventanas
     (p. ej. con −9h/−3h/−2h/−1h: hasta ~10 h desde ahora).
  3. Por cada partido, comprueba si estamos dentro de una ventana objetivo
     (p. ej. −9h ±5min, −3h ±5min, −2h ±5min, −1h ±5min por defecto).
  4. Si sí, lanza CADA scraper configurado en `CONFIG["scrapers"]` (modo fetch)
     secuencialmente — actualmente sólo 22bet (Codere temporalmente deshabilitado
     pendiente de re-bootstrap de endpoints; ver followup/restore-codere-scraper).
  5. Mantiene un `state.json` con qué capturas ya hicimos para no duplicar.
  6. Duerme 5 minutos y repite.

Se puede dejar corriendo indefinidamente en background (terminal o
Task Scheduler de Windows) — es barato en CPU/RAM.

Uso:
  python scripts/odds/odds_scheduler.py            # bucle infinito cada 5 min
  python scripts/odds/odds_scheduler.py --once     # 1 iteración (útil para cron/task)
  python scripts/odds/odds_scheduler.py --windows 9,3,2,1   # personaliza ventanas (horas)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.odds import db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("odds_scheduler.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("odds_scheduler")

CONFIG = {
    "calendar_path": "data/reference/liga_calendar_rows.csv",
    "state_path": "data/reference/odds_scheduler_state.json",
    # Lista de scrapers a invocar en cada ventana disparada. Cada entry es la
    # lista de args (sin el python interpreter — lo anteponemos en runtime).
    # Un fallo en uno NO bloquea a los demás (ver `_trigger_scraper()`).
    "scrapers": [
        ["scripts/odds/22bet_scraper.py"],
        # Codere: temporalmente deshabilitado — requiere re-bootstrap de
        # endpoints (human-interactive discovery via Playwright + edición
        # manual del JSON). Ver `followup/restore-codere-scraper` en engram.
        # Para re-habilitar, descomentar la línea siguiente una vez exista
        # `data/reference/codere_endpoints.json` con events_endpoint + markets_endpoint:
        # ["scripts/odds/codere_scraper.py"],
    ],
    "tz_local": "Europe/Madrid",  # kick-offs en hora peninsular
    "sleep_seconds": 300,         # 5 min
    "window_tolerance_min": 5,    # ±5 min sobre la hora objetivo
    "scraper_timeout_s": 300,     # timeout por scraper (cada uno, no global)
}


# ─────────────────────────────────────────────────────────────
# CALENDARIO
# ─────────────────────────────────────────────────────────────

def load_scheduled_next_hours(hours_ahead: float = 4.0) -> pd.DataFrame:
    """Devuelve partidos LaLiga con estado scheduled que empiezan en las
    próximas `hours_ahead` horas.
    """
    cal = pd.read_csv(CONFIG["calendar_path"])
    cal = cal[
        (cal["status"] == "scheduled")
        & (cal["competition"].str.contains("La Liga", case=False, na=False))
    ].copy()

    # Construir datetime en hora peninsular y pasar a UTC
    cal["match_time"] = cal["match_time"].fillna("00:00:00")
    cal["kickoff_local"] = pd.to_datetime(
        cal["match_date"].astype(str) + " " + cal["match_time"].astype(str),
        errors="coerce",
    )
    cal = cal.dropna(subset=["kickoff_local"])
    cal["kickoff_utc"] = (
        cal["kickoff_local"]
        .dt.tz_localize(CONFIG["tz_local"], ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    cal = cal[(cal["kickoff_utc"] > now) & (cal["kickoff_utc"] <= cutoff)]
    return cal.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if db.is_available():
        return db.load_state()
    p = Path(CONFIG["state_path"])
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    if db.is_available():
        db.save_state(state)
        return
    p = Path(CONFIG["state_path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _state_key(event_id: str, window_h: int) -> str:
    return f"{event_id}:t-{window_h}h"


# ─────────────────────────────────────────────────────────────
# DECIDIR Y LANZAR
# ─────────────────────────────────────────────────────────────

def _in_window(kickoff_utc: datetime, now: datetime, target_hours: int) -> bool:
    """True si `now` está en el intervalo [kickoff − target_hours ± tolerance]."""
    target_moment = kickoff_utc - timedelta(hours=target_hours)
    tol = timedelta(minutes=CONFIG["window_tolerance_min"])
    return target_moment - tol <= now <= target_moment + tol


def run_iteration(windows: list[int]) -> None:
    state = _load_state()
    cal = load_scheduled_next_hours(hours_ahead=max(windows) + 1)
    if cal.empty:
        log.info("Sin partidos en las próximas %d horas.", max(windows) + 1)
        return

    now = datetime.now(timezone.utc)
    triggered = False
    for _, row in cal.iterrows():
        ev_id = str(row["event_id"])
        kickoff = row["kickoff_utc"].to_pydatetime()
        for w in windows:
            key = _state_key(ev_id, w)
            if key in state:
                continue
            if _in_window(kickoff, now, w):
                log.info("Ventana T-%dh disparada: %s %s vs %s (kickoff %s UTC)",
                         w, ev_id, row["home_team"], row["away_team"],
                         kickoff.strftime("%Y-%m-%d %H:%M"))
                _trigger_scraper()
                state[key] = now.isoformat()
                triggered = True
                # No romper; queremos marcar todas las ventanas vencidas, pero
                # un solo fetch captura TODAS las cuotas de LaLiga a la vez.
                break
    if triggered:
        _save_state(state)


def _trigger_scraper() -> None:
    """Lanza TODOS los scrapers configurados en `CONFIG["scrapers"]` secuencialmente.

    Cada uno corre en modo fetch (sin Playwright, usa endpoints guardados o API
    pública según el scraper). Un fallo (exit != 0, excepción, timeout) en uno
    NO aborta los demás — logueamos WARN y seguimos. Esto garantiza que si Codere
    vuelve a fallar, 22bet igual captura, y viceversa.
    """
    scrapers = CONFIG.get("scrapers", [])
    if not scrapers:
        log.warning("CONFIG['scrapers'] vacío — nada que ejecutar.")
        return

    timeout = CONFIG.get("scraper_timeout_s", 300)
    for cmd_args in scrapers:
        # Nombre amigable del scraper = basename del script (último arg que termina en .py).
        script_name = next(
            (Path(a).name for a in cmd_args if str(a).endswith(".py")),
            " ".join(cmd_args),
        )
        full_cmd = [sys.executable, *cmd_args]
        log.info("Lanzando scraper: %s", script_name)
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                log.warning(
                    "Scraper %s devolvió código %d\nSTDOUT: %s\nSTDERR: %s",
                    script_name, result.returncode,
                    result.stdout[-500:], result.stderr[-500:],
                )
            else:
                log.info("Scraper %s OK (%d bytes stdout)",
                         script_name, len(result.stdout))
        except subprocess.TimeoutExpired:
            log.warning("Scraper %s timeout (>%ds) — continúo con el resto.",
                        script_name, timeout)
        except Exception as exc:
            log.error("Error ejecutando scraper %s: %s", script_name, exc)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true",
                        help="1 iteración y salir (para Task Scheduler/cron)")
    parser.add_argument("--windows", default="9,3,2,1",
                        help="Horas-antes-del-partido en que disparar (coma-separadas)")
    args = parser.parse_args()

    windows = sorted({int(w.strip()) for w in args.windows.split(",") if w.strip()})
    log.info("Ventanas objetivo: %s horas antes del kickoff", windows)
    db.init_tables()

    if args.once:
        run_iteration(windows)
        return

    log.info("Bucle infinito cada %ds. Ctrl+C para parar.", CONFIG["sleep_seconds"])
    try:
        while True:
            run_iteration(windows)
            time.sleep(CONFIG["sleep_seconds"])
    except KeyboardInterrupt:
        log.info("Detenido por usuario.")


if __name__ == "__main__":
    main()
