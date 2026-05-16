"""
22bet - Scraper de Saques de Banda - La Liga España
API: https://22play22.com/service-api/LineFeed/

Estructura descubierta (2026):
  - GetChampZip?champ=127733          -> partidos de La Liga (Value.G[])
  - GetGameZip?id={match}&isSubGames=true  -> subgames (Value.SG[])
        -> Saque de banda: TI=55 (TG='Saques de banda')
  - GetGameZip?id={subgame}           -> cuotas (Value.E[])

En las cuotas (Value.E):
  - GS=4, G=17 -> mercado Total (Over/Under)
      T=9  -> Over (Más)
      T=10 -> Under (Menos)
  - GS=1, G=1  -> mercado 1X2 (team_with_more, 3-way)
      T=1  -> home (equipo local con más saques)
      T=2  -> draw (empate en número de saques)
      T=3  -> away (equipo visitante con más saques)
  - P  -> línea (e.g. 41.5)   -- sólo aplica a O/U
  - C  -> cuota decimal (e.g. 1.95)

Esquema del parquet resultante (alineado con `odds_codere.parquet`):
  home_team, away_team, match_date, scraped_at, hours_before,
  bookmaker='22bet', market_type ('total_over_under'|'team_with_more'),
  line (float|NaN), side ('over'|'under'|'home'|'away'|'draw'),
  odds, raw_market_name, raw_selection, match_ci (traza)

Uso:
  python scripts/odds/22bet_scraper.py                         # fetch y guarda
  python scripts/odds/22bet_scraper.py --dry-run               # fetch sin guardar
  python scripts/odds/22bet_scraper.py --list-games            # lista partidos
  python scripts/odds/22bet_scraper.py --max-matches 3         # sólo N partidos (smoke)
  python scripts/odds/22bet_scraper.py --markets total_over_under  # un mercado
  python scripts/odds/22bet_scraper.py --markets all           # ambos (default)
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.odds import db

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
    "base_url": "https://22play22.com/service-api/LineFeed",
    "laliga_champ_id": 127733,
    "throw_in_ti": 55,
    # Mercado Total (Over/Under)
    "total_gs": 4,
    "total_g": 17,
    "over_t": 9,
    "under_t": 10,
    # Mercado 1X2 / team_with_more (3-way en saques)
    "twm_gs": 1,
    "twm_g": 1,
    "twm_home_t": 1,
    "twm_draw_t": 2,
    "twm_away_t": 3,
    # Salida
    "odds_history_path": "data/reference/odds_22bet.parquet",
    # Rate limiting cortés: jitter aleatorio alrededor de request_delay
    "request_delay": 1.5,       # base
    "request_jitter": 0.5,      # ±jitter
    "timeout": 20,
    "max_retries": 3,
    "retry_backoff": 2.0,       # factor exponencial
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


def _polite_sleep() -> None:
    """Duerme request_delay ± jitter aleatorio para evitar patrón de bot."""
    base = CONFIG["request_delay"]
    jit = CONFIG["request_jitter"]
    time.sleep(max(0.0, base + random.uniform(-jit, jit)))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": "https://22play22.com/",
    "Origin": "https://22play22.com",
}


def _get(endpoint: str, extra_params: dict | None = None) -> dict | None:
    """GET al endpoint de 22bet con retries + backoff exponencial.

    Sobre errores transitorios (timeout, 5xx, conexión) reintenta hasta
    `max_retries` veces con backoff `retry_backoff**attempt` segundos.
    """
    url = f"{CONFIG['base_url']}/{endpoint}"
    params = dict(CONFIG["params_base"])
    if extra_params:
        params.update(extra_params)

    last_exc: Exception | None = None
    for attempt in range(CONFIG["max_retries"]):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=CONFIG["timeout"])
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            # 4xx (excepto 429) no se reintentan: son errores del cliente
            if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                log.warning("HTTP %s en %s (no se reintenta)", status, url)
                return None
            last_exc = e
            log.info("HTTP %s en %s (intento %d/%d)", status, url, attempt + 1, CONFIG["max_retries"])
        except Exception as e:
            last_exc = e
            log.info("Error en %s: %s (intento %d/%d)", url, e, attempt + 1, CONFIG["max_retries"])
        if attempt < CONFIG["max_retries"] - 1:
            time.sleep(CONFIG["retry_backoff"] ** attempt)
    log.warning("Falló %s tras %d intentos: %s", url, CONFIG["max_retries"], last_exc)
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


def _get_subgame_events(subgame_ci: int) -> list[dict]:
    """Devuelve la lista Value.E (eventos/outcomes) de un subgame dado."""
    data = _get("GetGameZip", {
        "id": subgame_ci,
        "isSubGames": "false",
        "grMode": 4,
    })
    if not data:
        return []
    val = data.get("Value") or {}
    return val.get("E") or []


def _parse_total_events(events: list[dict]) -> list[dict]:
    """Extrae selecciones Over/Under (GS=4, G=17) de una lista de eventos."""
    out = []
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
        out.append({"market_type": "total_over_under",
                    "side": side,
                    "line": float(line),
                    "odds": float(price)})
    return out


def _parse_team_with_more_events(events: list[dict]) -> list[dict]:
    """Extrae selecciones 3-way (GS=1, G=1) home/draw/away. Línea = None."""
    t_to_side = {
        CONFIG["twm_home_t"]: "home",
        CONFIG["twm_draw_t"]: "draw",
        CONFIG["twm_away_t"]: "away",
    }
    out = []
    for ev in events:
        if ev.get("GS") != CONFIG["twm_gs"] or ev.get("G") != CONFIG["twm_g"]:
            continue
        side = t_to_side.get(ev.get("T"))
        if side is None:
            continue
        price = ev.get("C")
        if price is None:
            continue
        out.append({"market_type": "team_with_more",
                    "side": side,
                    "line": None,
                    "odds": float(price)})
    return out


def get_total_throwin_odds(subgame_ci: int) -> list[dict]:
    """[BACKWARD-COMPAT] extrae sólo O/U — mantenida por tests externos si los hay."""
    return _parse_total_events(_get_subgame_events(subgame_ci))


def _pick_best_subgame(match_ci: int) -> tuple[int | None, list[dict]]:
    """Selecciona el subgame (TI=55) con más outcomes Over/Under y devuelve
    sus eventos ya cacheados — evita hacer doble GET para O/U y team_with_more.

    Returns (subgame_ci, events). Si no hay subgame válido: (None, []).
    """
    cis = get_throw_in_subgames(match_ci)
    if not cis:
        return None, []

    best_ci: int | None = None
    best_count = 0
    best_events: list[dict] = []
    for ci in cis:
        _polite_sleep()
        events = _get_subgame_events(ci)
        # Criterio de ranking: cantidad de outcomes Over/Under (es el mercado
        # más rico y estable; el de 3-way suele existir junto a él)
        n_ou = sum(1 for e in events
                   if e.get("GS") == CONFIG["total_gs"] and e.get("G") == CONFIG["total_g"])
        if n_ou > best_count:
            best_count = n_ou
            best_ci = ci
            best_events = events
    return best_ci, best_events


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
    """Append-only a `odds_history_path`. Preserva histórico para trazabilidad
    temporal (como hace Codere). Concatena columnas union-style: si la versión
    vieja del parquet no tenía `market_type`/`raw_market_name`, las rellena con
    defaults razonables para que el merge funcione.
    """
    out = Path(CONFIG["odds_history_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = pd.read_parquet(out)
        # Migración suave: si el parquet viejo no tenía market_type,
        # asumimos que todas sus filas eran total_over_under (único mercado que
        # soportaba la versión anterior del scraper).
        if "market_type" not in prev.columns:
            prev["market_type"] = "total_over_under"
        if "raw_market_name" not in prev.columns:
            prev["raw_market_name"] = prev.get("market_name", "Saques de banda Total")
        if "raw_selection" not in prev.columns:
            prev["raw_selection"] = prev["side"].astype(str)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(out, index=False)
    log.info("odds_22bet actualizado: %s (%d filas totales)", out, len(df))


def _build_rows(
    game: dict,
    subgame_ci: int,
    events: list[dict],
    markets: tuple[str, ...],
    now_iso: str,
    start_iso: str | None,
    hours_before: float | None,
) -> list[dict]:
    """Construye filas unified-schema (alineadas con `odds_codere.parquet`)
    a partir de los eventos de 22bet de un subgame.

    Mercados incluidos según `markets`:
      - 'total_over_under' → Over/Under lines
      - 'team_with_more'   → 3-way home/draw/away (line=None)
    """
    rows: list[dict] = []

    if "total_over_under" in markets:
        for sel in _parse_total_events(events):
            rows.append({
                "match_ci": subgame_ci,          # traza del subgame 22bet
                "home_team": game["home"],
                "away_team": game["away"],
                "match_date": start_iso,
                "scraped_at": now_iso,
                "hours_before": round(hours_before, 2) if hours_before is not None else None,
                "market_type": "total_over_under",
                "line": sel["line"],
                "side": sel["side"],
                "odds": sel["odds"],
                "bookmaker": "22bet",
                "raw_market_name": "Saques de banda Total",
                "raw_selection": f"{'Más' if sel['side'] == 'over' else 'Menos'} de {sel['line']}",
                # Columna legacy mantenida para compat con el parquet viejo
                "market_name": "Saques de banda Total",
            })

    if "team_with_more" in markets:
        twm_selections = _parse_team_with_more_events(events)
        # Mapa side→team para raw_selection (traza)
        side_to_label = {
            "home": game["home"],
            "away": game["away"],
            "draw": "Empate",
        }
        for sel in twm_selections:
            rows.append({
                "match_ci": subgame_ci,
                "home_team": game["home"],
                "away_team": game["away"],
                "match_date": start_iso,
                "scraped_at": now_iso,
                "hours_before": round(hours_before, 2) if hours_before is not None else None,
                "market_type": "team_with_more",
                "line": None,
                "side": sel["side"],
                "odds": sel["odds"],
                "bookmaker": "22bet",
                "raw_market_name": "Equipo con más saques de banda",
                "raw_selection": side_to_label.get(sel["side"], sel["side"]),
                "market_name": "Equipo con más saques de banda",
            })

    return rows


def fetch_odds(
    dry_run: bool = False,
    markets: tuple[str, ...] = ("total_over_under", "team_with_more"),
    max_matches: int | None = None,
    match_date_filter: str | None = None,
) -> pd.DataFrame:
    """Fetch de cuotas 22bet para saques de banda en La Liga.

    Parámetros:
      dry_run:        Si True, no guarda en parquet (sólo stdout).
      markets:        Subset a extraer ('total_over_under', 'team_with_more').
      max_matches:    Límite de partidos a procesar (útil para smoke tests).
      match_date_filter: 'YYYYMMDD' — sólo partidos cuyo kick-off UTC cae en
                      esa fecha. None = todos los upcoming.
    """
    games = get_laliga_games()
    if not games:
        log.warning("No se encontraron partidos de La Liga en 22bet.")
        return pd.DataFrame()

    # Filtro por fecha (si se pidió)
    if match_date_filter:
        def _in_date(g: dict) -> bool:
            iso = _unix_to_iso(g["start_unix"])
            if not iso:
                return False
            try:
                return pd.to_datetime(iso, utc=True).strftime("%Y%m%d") == match_date_filter
            except Exception:
                return False
        games = [g for g in games if _in_date(g)]
        log.info("Filtro fecha=%s → %d partidos restantes", match_date_filter, len(games))

    if max_matches is not None:
        games = games[:max_matches]
        log.info("max_matches=%d → procesando sólo los primeros %d partidos",
                 max_matches, len(games))

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    rows: list[dict] = []

    # Contadores para log final (ok/missing/error por partido)
    stats = {"ok": 0, "missing": 0, "error": 0}

    for game in games:
        ci = game["ci"]
        home = game["home"]
        away = game["away"]
        start_iso = _unix_to_iso(game["start_unix"])
        hours_before = _hours_before(now, start_iso)
        h_str = f"{hours_before:.1f}h" if hours_before is not None else "?"

        log.info("Procesando: %s vs %s (CI=%s, ~%s antes del inicio)",
                 home, away, ci, h_str)

        _polite_sleep()
        try:
            subgame_ci, events = _pick_best_subgame(ci)
        except Exception as exc:
            log.warning("  -> [error] %s vs %s: %s", home, away, exc)
            stats["error"] += 1
            continue

        if subgame_ci is None or not events:
            log.info("  -> [missing] Sin subgame de saques de banda con mercado Total")
            stats["missing"] += 1
            continue

        match_rows = _build_rows(
            game, subgame_ci, events, markets, now_iso, start_iso, hours_before,
        )
        if not match_rows:
            log.info("  -> [missing] Subgame %s sin cuotas parseables en mercados %s",
                     subgame_ci, markets)
            stats["missing"] += 1
            continue

        rows.extend(match_rows)
        stats["ok"] += 1
        # Resumen por mercado
        n_ou  = sum(1 for r in match_rows if r["market_type"] == "total_over_under")
        n_twm = sum(1 for r in match_rows if r["market_type"] == "team_with_more")
        ou_lines = [r["line"] for r in match_rows if r["market_type"] == "total_over_under"]
        line_range = f"{min(ou_lines)}-{max(ou_lines)}" if ou_lines else "n/a"
        log.info("  -> [ok] O/U: %d cuotas (líneas %s) | team_with_more: %d cuotas",
                 n_ou, line_range, n_twm)

    log.info("Resumen por partido: ok=%d missing=%d error=%d",
             stats["ok"], stats["missing"], stats["error"])

    df = pd.DataFrame(rows)

    # Orden determinista (invariante de idempotencia del proyecto)
    if len(df):
        sort_cols = [c for c in ["match_ci", "market_type", "side", "line"] if c in df.columns]
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    n_matches = df["match_ci"].nunique() if len(df) else 0
    log.info("Total: %d cuotas extraídas para %d partidos", len(df), n_matches)

    if dry_run:
        if len(df):
            print(df.to_string(index=False))
        else:
            print("Sin cuotas disponibles en este momento.")
        return df

    if len(df):
        _append_to_history(df)
        db.upsert_odds(df)

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
    parser.add_argument("--markets", default="all",
                        choices=["all", "total_over_under", "team_with_more"],
                        help="Qué mercado extraer (default: all)")
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Límite de partidos a procesar (útil para smoke)")
    parser.add_argument("--date", default=None,
                        help="Filtrar partidos con kick-off en YYYYMMDD (UTC)")
    parser.add_argument("--all-upcoming", action="store_true",
                        help="Todos los partidos futuros (default — kept para paridad CLI)")
    parser.add_argument("--headed", action="store_true",
                        help="[no-op] mantenido por paridad con scrapers Playwright; "
                             "este scraper usa HTTP puro.")
    args = parser.parse_args()

    if args.headed:
        log.info("--headed es no-op en este scraper (usa HTTP, no navegador).")

    if args.list_games:
        list_games()
        return

    markets = ("total_over_under", "team_with_more") if args.markets == "all" else (args.markets,)
    fetch_odds(
        dry_run=args.dry_run,
        markets=markets,
        max_matches=args.max_matches,
        match_date_filter=args.date,
    )


if __name__ == "__main__":
    main()
