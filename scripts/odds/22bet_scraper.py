"""
22bet - Scraper de Saques de Banda - La Liga España
API: https://22bet92.com/service-api/LineFeed/

Estructura descubierta (2026):
  - GetChampZip?champ=127733          -> partidos de La Liga (Value.G[])
  - GetGameZip?id={match}&isSubGames=true  -> subgames (Value.SG[])
        -> Saque de banda: TI=55 (TG='Saques de banda')
  - GetGameZip?id={subgame}           -> cuotas (Value.E[])

En las cuotas (Value.E):
  - GS=4, G=17 -> mercado Total (Over/Under)
      T=9  -> Over (Más)
      T=10 -> Under (Menos)
  - P  -> línea (e.g. 41.5)
  - C  -> cuota decimal (e.g. 1.95)

Uso:
  python scripts/odds/22bet_scraper.py                # fetch y guarda
  python scripts/odds/22bet_scraper.py --dry-run      # fetch sin guardar
  python scripts/odds/22bet_scraper.py --list-games   # lista partidos de La Liga
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("22bet_scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("22bet_scraper")

CONFIG = {
    "base_url": "https://22bet92.com/service-api/LineFeed",
    "laliga_champ_id": 127733,
    "throw_in_ti": 55,
    "total_gs": 4,
    "total_g": 17,
    "over_t": 9,
    "under_t": 10,
    "odds_history_path": "data/reference/odds_22bet.parquet",
    "request_delay": 0.4,
    "timeout": 20,
    "params_base": {
        "lng": "es_ES",
        "tf": 3000000,
        "tz": 2,
        "country": 78,
        "partner": 151,
        "gr": 151,
        "mode": 4,
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": "https://22bet92.com/",
    "Origin": "https://22bet92.com",
}


def _get(endpoint: str, extra_params: dict | None = None) -> dict | None:
    url = f"{CONFIG['base_url']}/{endpoint}"
    params = dict(CONFIG["params_base"])
    if extra_params:
        params.update(extra_params)
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=CONFIG["timeout"])
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        log.warning("HTTP %s en %s", e.response.status_code, url)
    except Exception as e:
        log.warning("Error en %s: %s", url, e)
    return None


def get_laliga_games() -> list[dict]:
    """Devuelve partidos próximos de La Liga (champ 127733)."""
    data = _get("GetChampZip", {
        "champ": CONFIG["laliga_champ_id"],
        "groupChamps": "true",
    })
    if not data:
        return []

    val = data.get("Value") or {}
    games_raw = val.get("G") if isinstance(val, dict) else []
    if not games_raw:
        log.warning("GetChampZip sin partidos. Value keys: %s", list(val.keys()) if isinstance(val, dict) else type(val))
        return []

    games = []
    for g in games_raw:
        if g.get("LI") and g["LI"] != CONFIG["laliga_champ_id"]:
            continue
        home = g.get("O1", "")
        away = g.get("O2", "")
        if home.lower() in {"locales", "visitantes"} or away.lower() in {"locales", "visitantes"}:
            continue
        games.append({
            "ci": g.get("CI"),
            "home": home,
            "away": away,
            "start_unix": g.get("S"),
            "league": g.get("LE", ""),
        })
    log.info("Partidos de La Liga encontrados: %d", len(games))
    return games


def get_throw_in_subgames(match_ci: int) -> list[int]:
    """Devuelve los CI de los subgames de 'Saques de banda' (TI=55)."""
    data = _get("GetGameZip", {
        "id": match_ci,
        "isSubGames": "true",
        "grMode": 4,
    })
    if not data:
        return []

    val = data.get("Value") or {}
    sg_list = val.get("SG") or []

    cis = []
    for s in sg_list:
        if s.get("TI") == CONFIG["throw_in_ti"]:
            ci = s.get("CI")
            if ci is not None:
                cis.append(ci)
    return cis


def get_total_throwin_odds(subgame_ci: int) -> list[dict]:
    """Extrae cuotas Over/Under del mercado Total (GS=4) para saques de banda."""
    data = _get("GetGameZip", {
        "id": subgame_ci,
        "isSubGames": "false",
        "grMode": 4,
    })
    if not data:
        return []

    val = data.get("Value") or {}
    events = val.get("E") or []

    selections = []
    for ev in events:
        if ev.get("GS") != CONFIG["total_gs"] or ev.get("G") != CONFIG["total_g"]:
            continue
        t = ev.get("T")
        if t == CONFIG["over_t"]:
            side = "over"
        elif t == CONFIG["under_t"]:
            side = "under"
        else:
            continue
        line = ev.get("P")
        price = ev.get("C")
        if line is None or price is None:
            continue
        selections.append({
            "side": side,
            "line": float(line),
            "odds": float(price),
        })
    return selections


def _pick_total_subgame(match_ci: int) -> int | None:
    """Selecciona el subgame de saques de banda que tiene mercado Total.
    Si hay varios, elige el que tenga más outcomes de Total.
    """
    cis = get_throw_in_subgames(match_ci)
    if not cis:
        return None

    best_ci = None
    best_count = 0
    for ci in cis:
        time.sleep(CONFIG["request_delay"])
        odds = get_total_throwin_odds(ci)
        if len(odds) > best_count:
            best_count = len(odds)
            best_ci = ci
    return best_ci


def _unix_to_iso(s: int | None) -> str | None:
    if s is None:
        return None
    try:
        s_int = int(s)
    except (TypeError, ValueError):
        return None
    if s_int > 10_000_000_000:  # ms
        s_int //= 1000
    try:
        return datetime.fromtimestamp(s_int, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _hours_before(now: datetime, start_iso: str | None) -> float | None:
    if not start_iso:
        return None
    try:
        start = pd.to_datetime(start_iso, utc=True).to_pydatetime()
        return (start - now).total_seconds() / 3600.0
    except Exception:
        return None


def _append_to_history(df: pd.DataFrame) -> None:
    out = Path(CONFIG["odds_history_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = pd.read_parquet(out)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(out, index=False)
    log.info("odds_22bet actualizado: %s (%d filas totales)", out, len(df))


def fetch_odds(dry_run: bool = False) -> pd.DataFrame:
    games = get_laliga_games()
    if not games:
        log.warning("No se encontraron partidos de La Liga en 22bet.")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    rows = []

    for game in games:
        ci = game["ci"]
        home = game["home"]
        away = game["away"]
        start_iso = _unix_to_iso(game["start_unix"])
        hours_before = _hours_before(now, start_iso)
        h_str = f"{hours_before:.1f}h" if hours_before is not None else "?"

        log.info("Procesando: %s vs %s (CI=%s, ~%s antes del inicio)",
                 home, away, ci, h_str)

        time.sleep(CONFIG["request_delay"])
        subgame_ci = _pick_total_subgame(ci)

        if subgame_ci is None:
            log.info("  -> Sin mercado de saques de banda disponible")
            continue

        selections = get_total_throwin_odds(subgame_ci)
        if not selections:
            log.info("  -> Subgame %s encontrado pero sin cuotas de Total", subgame_ci)
            continue

        for sel in selections:
            rows.append({
                "match_ci": ci,
                "home_team": home,
                "away_team": away,
                "match_date": start_iso,
                "scraped_at": now.isoformat(),
                "hours_before": round(hours_before, 2) if hours_before is not None else None,
                "market_name": "Saques de banda Total",
                "line": sel["line"],
                "side": sel["side"],
                "odds": sel["odds"],
                "bookmaker": "22bet",
            })

        log.info("  -> %d cuotas extraídas (líneas %s-%s)",
                 len(selections),
                 min(s["line"] for s in selections),
                 max(s["line"] for s in selections))

    df = pd.DataFrame(rows)
    log.info("Total: %d cuotas extraídas para %d partidos",
             len(df), df["match_ci"].nunique() if len(df) else 0)

    if dry_run:
        if len(df):
            print(df.to_string(index=False))
        else:
            print("Sin cuotas disponibles en este momento.")
        return df

    if len(df):
        _append_to_history(df)

    return df


def list_games() -> None:
    games = get_laliga_games()
    if not games:
        print("No hay partidos disponibles.")
        return
    for g in games:
        start_iso = _unix_to_iso(g["start_unix"])
        print(f"CI={g['ci']:>11}  {g['home'][:28]:<28} vs {g['away'][:28]:<28}  {start_iso or 'sin fecha'}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="22bet scraper - saques de banda La Liga")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch sin guardar a parquet (debug)")
    parser.add_argument("--list-games", action="store_true",
                        help="Lista partidos activos de La Liga y sus CI")
    args = parser.parse_args()

    if args.list_games:
        list_games()
    else:
        fetch_odds(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
