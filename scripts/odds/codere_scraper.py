"""
Codere - Scraper de Saques de Banda - La Liga España
=====================================================
API pública: `https://m.apuestas.codere.es/NavigationService/*`

Endpoints descubiertos (Apr 2026, change `automate-codere-discovery`):

  1. `Event/GetEvents?parentId={league_node_id}`
       -> lista de partidos próximos de una competición.
       Para LaLiga (`Primera División`) `parentId = 2903511051`.
       Devuelve dicts con `NodeId`, `ParticipantHome`, `ParticipantAway`,
       `StartDate` (formato `/Date(millis)/`), etc.

  2. `Category/GetCategoryNoLiveInfos?parentid={event_node_id}`
       -> lista de categorías de mercados para un evento
       (`PRINCIPALES`, `ESTADÍSTICAS`, `TIROS`, ...).

  3. `Game/GetGamesNoLiveByCategoryInfo?parentid={event_node_id}&categoryInfoId={cat_id}`
       -> lista de mercados (`Games`) con sus `Results` (selecciones + cuotas).

En `Results`:
  - `Odd`           -> cuota decimal (e.g. 1.95)
  - `Name`          -> label selección ("Más de 41.5", "Menos de 41.5",
                       "Real Betis", "Empate", etc.)
  - `GameSpecialOddsValue` en el padre (`Spov`) -> línea cuando el mercado es
                       O/U ("`<Spov>41.5`"), vacío en mercados 3-way.

Mercados objetivo (nombres exactos usados en el histórico
`data/reference/odds_codere.parquet`):
  - "Total de Saques de Banda Más/Menos"          -> `total_over_under`
  - "Equipo con Más Saques de Banda"              -> `team_with_more`

Nota importante — VENTANA DE DISPONIBILIDAD:
  Codere suele abrir el mercado de saques de banda SÓLO en las horas previas
  al kick-off (no es pre-match permanente). Si la API no devuelve mercados de
  saques ahora mismo, el scraper sale con 0 filas y exit 0 (success) — el
  scheduler lo volverá a llamar en la siguiente ventana T-3h/-2h/-1h.

Esquema del parquet resultante (alineado con `odds_codere.parquet` legacy
y con `odds_22bet.parquet`):
  home_team, away_team, scraped_at,
  bookmaker='codere', market_type ('total_over_under'|'team_with_more'),
  line (float|NaN), side ('over'|'under'|'home'|'away'|'draw'),
  odds, raw_market_name, raw_selection

Uso:
  python scripts/odds/codere_scraper.py                         # fetch y guarda
  python scripts/odds/codere_scraper.py --dry-run               # fetch sin guardar
  python scripts/odds/codere_scraper.py --list-games            # lista partidos
  python scripts/odds/codere_scraper.py --max-matches 3         # sólo N partidos (smoke)
  python scripts/odds/codere_scraper.py --markets total_over_under  # un mercado
  python scripts/odds/codere_scraper.py --markets all           # ambos (default)
  python scripts/odds/codere_scraper.py --force-rediscover      # refresca endpoints cache

Discovery automático:
  El scraper usa rutas API fijas. La primera corrida (o cuando la cache expira
  a los 7 días) ping-ea cada endpoint para verificar que sigue respondiendo
  200 y persiste `data/reference/codere_endpoints.json` como cache/traza para
  auditoría externa. NO requiere Playwright ni interacción humana.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("codere_scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("codere_scraper")

CONFIG = {
    "base_url": "https://m.apuestas.codere.es/NavigationService",
    "laliga_node_id": 2903511051,  # Primera División

    # Paths descubiertos (ver docstring).
    "events_path": "Event/GetEvents",
    "categories_path": "Category/GetCategoryNoLiveInfos",
    "games_path": "Game/GetGamesNoLiveByCategoryInfo",

    # Cache de endpoints — refresca cada N días para detectar si algún path
    # devuelve 404 (ruptura de API) sin rehacer todo el scrape. No contiene
    # secretos ni cookies: es sólo traza.
    "endpoints_path": "data/reference/codere_endpoints.json",
    "endpoints_ttl_days": 7,

    # Output
    "odds_history_path": "data/reference/odds_codere.parquet",

    # Rate limiting + retries
    "request_delay": 1.5,    # base
    "request_jitter": 0.5,   # ±jitter
    "timeout": 20,
    "max_retries": 3,
    "retry_backoff": 2.0,    # factor exponencial

    # Keywords de matching de mercado (lowercase, case-insensitive). Cubre
    # tanto "saques de banda" como variantes de otros idiomas defensivo.
    "throw_in_keywords": ["saque", "banda", "throw-in", "throw in"],

    # Categorías prioritarias donde Codere listó históricamente 'Saques de
    # Banda' (ESTADÍSTICAS=78, ESPECIALES=60, PRINCIPALES=99, EQUIPOS=40).
    # Mantenemos este subset para evitar pedir las 19 categorías × N partidos
    # en cada fetch — reduce tiempo de scrape de ~5min a ~1min. Si el mercado
    # se ofrece en una categoría no listada, el scrape no la verá, pero ese
    # set cubre el 100% del histórico observado.
    "priority_category_ids": [78, 60, 99, 40],
}


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Referer": "https://m.apuestas.codere.es/deportes/",
    "Origin": "https://m.apuestas.codere.es",
}


def _polite_sleep() -> None:
    """Duerme request_delay ± jitter aleatorio para evitar patrón de bot."""
    base = CONFIG["request_delay"]
    jit = CONFIG["request_jitter"]
    time.sleep(max(0.0, base + random.uniform(-jit, jit)))


# ─────────────────────────────────────────────────────────────
# HTTP layer (mirror del patrón 22bet)
# ─────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> list | dict | None:
    """GET al endpoint de Codere con retries + backoff exponencial.

    `path` es relativo a `CONFIG["base_url"]`. Parámetros via querystring.
    Devuelve el JSON parseado, o None si todos los reintentos fallan.
    """
    url = f"{CONFIG['base_url']}/{path}"
    last_exc: Exception | None = None
    for attempt in range(CONFIG["max_retries"]):
        try:
            r = requests.get(url, params=params or {}, headers=HEADERS,
                             timeout=CONFIG["timeout"])
            r.raise_for_status()
            # Codere API responde 200 con cuerpo vacío si no hay data; cubrir.
            if not r.content:
                return None
            return r.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if (e.response is not None and 400 <= e.response.status_code < 500
                    and e.response.status_code != 429):
                log.warning("HTTP %s en %s (no se reintenta)", status, url)
                return None
            last_exc = e
            log.info("HTTP %s en %s (intento %d/%d)", status, url,
                     attempt + 1, CONFIG["max_retries"])
        except Exception as e:
            last_exc = e
            log.info("Error en %s: %s (intento %d/%d)", url, e,
                     attempt + 1, CONFIG["max_retries"])
        if attempt < CONFIG["max_retries"] - 1:
            time.sleep(CONFIG["retry_backoff"] ** attempt)
    log.warning("Falló %s tras %d intentos: %s", url, CONFIG["max_retries"], last_exc)
    return None


# ─────────────────────────────────────────────────────────────
# Endpoints cache (auto-discovery)
# ─────────────────────────────────────────────────────────────

def _endpoints_cache_fresh() -> bool:
    """True si el JSON de cache existe, tiene las claves requeridas y no ha
    expirado. Si no, hace falta regenerarlo."""
    p = Path(CONFIG["endpoints_path"])
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = {"events_endpoint", "markets_endpoint", "discovered_at"}
    if not required.issubset(data):
        return False
    try:
        ts = datetime.fromisoformat(data["discovered_at"].replace("Z", "+00:00"))
    except Exception:
        return False
    age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0
    return age_days <= CONFIG["endpoints_ttl_days"]


def _discover_endpoints(force: bool = False) -> dict:
    """Verifica las 3 rutas API contra un partido real de LaLiga y persiste
    `codere_endpoints.json` con `events_endpoint` + `markets_endpoint` +
    `discovered_at` (+ sample_response_shapes).

    100% headless, 100% automático — sin Playwright. La función pinguea cada
    endpoint; si uno devuelve 404 o no está bien formado, retorna un dict
    `{"ok": False, "error": "..."}` y NO escribe el archivo.
    """
    if not force and _endpoints_cache_fresh():
        p = Path(CONFIG["endpoints_path"])
        return json.loads(p.read_text(encoding="utf-8"))

    log.info("Verificando endpoints Codere (discovery automático)...")
    league = CONFIG["laliga_node_id"]

    events = _get(CONFIG["events_path"], {"parentId": league, "gameTypes": ""})
    if not isinstance(events, list) or not events:
        return {"ok": False,
                "error": "events endpoint no devolvió lista de partidos"}

    sample_event = events[0]
    eid = sample_event.get("NodeId")
    if not eid:
        return {"ok": False, "error": "event sin NodeId"}

    cats = _get(CONFIG["categories_path"], {"parentid": eid})
    if not isinstance(cats, list):
        return {"ok": False, "error": "categories endpoint no devolvió lista"}

    # Pinguea la primera categoría para validar el path de games.
    cat_id = cats[0].get("CategoryId") if cats else None
    games = _get(CONFIG["games_path"],
                 {"parentid": eid, "categoryInfoId": cat_id}) if cat_id else None
    if games is not None and not isinstance(games, list):
        return {"ok": False, "error": "games endpoint no devolvió lista"}

    out = {
        "discovered_at": datetime.now(timezone.utc).isoformat(),
        "discovered_via": "codere_scraper._discover_endpoints (automatic HTTP)",
        "base_url": CONFIG["base_url"],
        # Campos canónicos esperados por consumers externos (smokes, followups):
        "events_endpoint": (
            f"{CONFIG['base_url']}/{CONFIG['events_path']}"
            f"?parentId={{league_node_id}}&gameTypes="
        ),
        "markets_endpoint": (
            f"{CONFIG['base_url']}/{CONFIG['games_path']}"
            f"?parentid={{event_node_id}}&categoryInfoId={{category_id}}"
        ),
        "categories_endpoint": (
            f"{CONFIG['base_url']}/{CONFIG['categories_path']}"
            f"?parentid={{event_node_id}}"
        ),
        "laliga_node_id": league,
        "sample_response_shapes": {
            "events_first_item_keys": list(sample_event.keys())[:20],
            "n_events_today": len(events),
            "n_categories_first_event": len(cats),
            "n_games_first_category": len(games or []),
        },
    }

    outp = Path(CONFIG["endpoints_path"])
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Endpoints verificados + persistidos en %s "
             "(events=%d, cats=%d, games_sample=%d)",
             outp, len(events), len(cats), len(games or []))
    return out


# ─────────────────────────────────────────────────────────────
# Data layer
# ─────────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"/Date\((-?\d+)\)/")


def _parse_codere_date(raw: str | None) -> str | None:
    """Parsea el formato raro `/Date(1777057200000)/` a ISO-8601 UTC."""
    if not raw:
        return None
    m = _DATE_RE.search(raw)
    if not m:
        return None
    try:
        ms = int(m.group(1))
    except ValueError:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def get_laliga_games() -> list[dict]:
    """Devuelve partidos próximos de La Liga (Primera División)."""
    data = _get(CONFIG["events_path"],
                {"parentId": CONFIG["laliga_node_id"], "gameTypes": ""})
    if not isinstance(data, list):
        log.warning("GetEvents no devolvió lista: %r", type(data))
        return []

    games = []
    for e in data:
        eid = e.get("NodeId")
        home = e.get("ParticipantHome")
        away = e.get("ParticipantAway")
        if not (eid and home and away):
            continue
        games.append({
            "event_id": eid,
            "home": home,
            "away": away,
            "start_iso": _parse_codere_date(e.get("StartDate")),
            "league": e.get("LeagueName", ""),
            "children_count": e.get("ChildrenCount"),
        })
    log.info("Partidos de La Liga encontrados: %d", len(games))
    return games


def _get_event_categories(event_id: str) -> list[dict]:
    data = _get(CONFIG["categories_path"], {"parentid": event_id})
    return data if isinstance(data, list) else []


def _get_games_for_category(event_id: str, category_id: int | str) -> list[dict]:
    data = _get(CONFIG["games_path"],
                {"parentid": event_id, "categoryInfoId": category_id})
    return data if isinstance(data, list) else []


def _match_throw_in_name(name: str) -> bool:
    """Heurística permisiva — Codere usa 'Total de Saques de Banda Más/Menos'
    y 'Equipo con Más Saques de Banda'. Coincidencia por keywords lower-case.
    """
    low = (name or "").lower()
    return "saque" in low and "banda" in low


def _extract_line_from_spov(spov: str | None) -> float | None:
    """El campo `Spov` del game trae la línea como '<Spov>41.5'. Extrae el
    número. Devuelve None si no aplica (3-way market)."""
    if not spov:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", spov)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _classify_ou_selection(result_name: str) -> str | None:
    """'Más de 41.5' -> over | 'Menos de 41.5' -> under."""
    low = (result_name or "").lower()
    if "más" in low or "mas" in low or "over" in low:
        return "over"
    if "menos" in low or "under" in low:
        return "under"
    return None


def _classify_twm_selection(result_name: str, home: str, away: str) -> str | None:
    """Para 'Equipo con Más Saques de Banda': 3 selecciones = home team name,
    away team name, y 'Empate' (draw). Clasifica por match de nombre.
    """
    low = (result_name or "").lower()
    if low in {"empate", "draw", "x"} or "empate" in low:
        return "draw"
    # Match por prefijo (Codere puede devolver 'Real Betis' exacto, o variantes)
    if home and home.lower() in low:
        return "home"
    if away and away.lower() in low:
        return "away"
    # Si no matchea por nombre, como fallback usar orden (home primero)
    return None


def _find_throw_in_games(event_id: str) -> list[dict]:
    """Recorre las categorías prioritarias del evento (ver
    `CONFIG["priority_category_ids"]`) y filtra games cuyo Name contiene
    'saque' + 'banda'.

    Optimización: en lugar de las 19 categorías totales, sólo pedimos el
    subset donde Codere suele colocar este mercado. Reduce wall-clock de
    ~5min/partido a ~20s/partido cuando no hay mercado (caso mayoritario
    lejos del kick-off).
    """
    cats = _get_event_categories(event_id)
    if not cats:
        return []
    priority = set(CONFIG.get("priority_category_ids") or [])
    # Preserve category order but filter to priority (+ always include
    # IsRelevant=True for futureproofing — Codere marks PRINCIPALES relevant).
    target_cats = [c for c in cats
                   if c.get("CategoryId") in priority or c.get("IsRelevant")]
    found = []
    for c in target_cats:
        cat_id = c.get("CategoryId")
        if cat_id is None:
            continue
        _polite_sleep()
        games = _get_games_for_category(event_id, cat_id)
        for g in games:
            if _match_throw_in_name(g.get("Name", "")):
                found.append(g)
    return found


def _build_rows(
    game_meta: dict,
    throwin_games: list[dict],
    markets: tuple[str, ...],
    now_iso: str,
) -> list[dict]:
    """Construye filas unified-schema a partir de los `games` (mercados) de
    saques de banda de un partido.
    """
    rows: list[dict] = []
    home = game_meta["home"]
    away = game_meta["away"]

    for g in throwin_games:
        name = g.get("Name") or ""
        name_low = name.lower()
        spov = g.get("Spov") or ""
        line = _extract_line_from_spov(spov)
        results = g.get("Results") or []

        # O/U: nombre tipo "Total de Saques de Banda Más/Menos", con línea en Spov
        is_over_under = ("más/menos" in name_low or "mas/menos" in name_low
                         or "más\u002fmenos" in name_low or line is not None)
        is_team_with_more = ("equipo con más" in name_low
                             or "equipo con mas" in name_low)

        if is_over_under and "total_over_under" in markets and line is not None:
            for r in results:
                sel_name = r.get("Name") or ""
                side = _classify_ou_selection(sel_name)
                price = r.get("Odd")
                if side is None or price is None:
                    continue
                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "scraped_at": now_iso,
                    "bookmaker": "codere",
                    "market_type": "total_over_under",
                    "line": float(line),
                    "side": side,
                    "odds": float(price),
                    "raw_market_name": name,
                    "raw_selection": sel_name,
                })

        if is_team_with_more and "team_with_more" in markets:
            for r in results:
                sel_name = r.get("Name") or ""
                side = _classify_twm_selection(sel_name, home, away)
                price = r.get("Odd")
                if side is None or price is None:
                    continue
                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "scraped_at": now_iso,
                    "bookmaker": "codere",
                    "market_type": "team_with_more",
                    "line": None,
                    "side": side,
                    "odds": float(price),
                    "raw_market_name": name,
                    "raw_selection": sel_name,
                })

    return rows


def _append_to_history(df: pd.DataFrame) -> None:
    """Append-only a `odds_history_path`. Preserva histórico para trazabilidad
    temporal. Idempotente respecto al schema (match schema del parquet legacy).
    """
    out = Path(CONFIG["odds_history_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = pd.read_parquet(out)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(out, index=False)
    log.info("odds_codere actualizado: %s (%d filas totales)", out, len(df))


# ─────────────────────────────────────────────────────────────
# Main fetch
# ─────────────────────────────────────────────────────────────

def fetch_odds(
    dry_run: bool = False,
    markets: tuple[str, ...] = ("total_over_under", "team_with_more"),
    max_matches: int | None = None,
    force_rediscover: bool = False,
) -> pd.DataFrame:
    """Fetch de cuotas Codere para saques de banda en La Liga.

    Parámetros:
      dry_run:          Si True, no guarda a parquet (sólo stdout).
      markets:          Subset a extraer ('total_over_under', 'team_with_more').
      max_matches:      Límite de partidos a procesar (útil para smoke tests).
      force_rediscover: Forzar refresh de la cache de endpoints.
    """
    discovery = _discover_endpoints(force=force_rediscover)
    if not discovery.get("events_endpoint"):
        log.error("Discovery falló: %s", discovery.get("error", "desconocido"))
        return pd.DataFrame()

    games = get_laliga_games()
    if not games:
        log.warning("No hay partidos de La Liga en Codere.")
        return pd.DataFrame()

    if max_matches is not None:
        games = games[:max_matches]
        log.info("max_matches=%d → procesando sólo los primeros %d partidos",
                 max_matches, len(games))

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    rows: list[dict] = []
    stats = {"ok": 0, "missing": 0, "error": 0}

    for game in games:
        home, away, eid = game["home"], game["away"], game["event_id"]
        start_iso = game["start_iso"]
        hours_before = None
        if start_iso:
            try:
                start = pd.to_datetime(start_iso, utc=True).to_pydatetime()
                hours_before = (start - now).total_seconds() / 3600.0
            except Exception:
                pass

        h_str = f"{hours_before:.1f}h" if hours_before is not None else "?"
        log.info("Procesando: %s vs %s (eid=%s, ~%s antes del inicio)",
                 home, away, eid, h_str)

        _polite_sleep()
        try:
            throwin_games = _find_throw_in_games(eid)
        except Exception as exc:
            log.warning("  -> [error] %s vs %s: %s", home, away, exc)
            stats["error"] += 1
            continue

        if not throwin_games:
            log.info("  -> [missing] Sin mercados de saques de banda")
            stats["missing"] += 1
            continue

        match_rows = _build_rows(game, throwin_games, markets, now_iso)
        if not match_rows:
            log.info("  -> [missing] Mercados de saques presentes pero sin "
                     "selecciones parseables")
            stats["missing"] += 1
            continue

        rows.extend(match_rows)
        stats["ok"] += 1
        n_ou = sum(1 for r in match_rows if r["market_type"] == "total_over_under")
        n_twm = sum(1 for r in match_rows if r["market_type"] == "team_with_more")
        ou_lines = sorted({r["line"] for r in match_rows
                           if r["market_type"] == "total_over_under"})
        line_str = (f"{ou_lines[0]}-{ou_lines[-1]} ({len(ou_lines)} líneas)"
                    if ou_lines else "n/a")
        log.info("  -> [ok] O/U: %d cuotas (líneas %s) | "
                 "team_with_more: %d cuotas", n_ou, line_str, n_twm)

    log.info("Resumen por partido: ok=%d missing=%d error=%d",
             stats["ok"], stats["missing"], stats["error"])

    df = pd.DataFrame(rows)

    # Orden determinista
    if len(df):
        sort_cols = [c for c in
                     ["home_team", "away_team", "market_type", "side", "line"]
                     if c in df.columns]
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    n_matches = df[["home_team", "away_team"]].drop_duplicates().shape[0] if len(df) else 0
    log.info("Total: %d cuotas extraídas para %d partidos", len(df), n_matches)

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
    _discover_endpoints()
    games = get_laliga_games()
    if not games:
        print("No hay partidos disponibles.")
        return
    for g in games:
        print(f"eid={g['event_id']:>11}  "
              f"{g['home'][:28]:<28} vs {g['away'][:28]:<28}  "
              f"{g['start_iso'] or 'sin fecha'}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Codere scraper - saques de banda La Liga")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch sin guardar a parquet (debug)")
    parser.add_argument("--list-games", action="store_true",
                        help="Lista partidos activos de La Liga y sus NodeId")
    parser.add_argument("--markets", default="all",
                        choices=["all", "total_over_under", "team_with_more"],
                        help="Qué mercado extraer (default: all)")
    parser.add_argument("--max-matches", type=int, default=None,
                        help="Límite de partidos a procesar (útil para smoke)")
    parser.add_argument("--force-rediscover", action="store_true",
                        help="Ignora la cache de endpoints y pinguea de nuevo")
    # --discover se mantiene como alias por compat histórica con el scraper
    # anterior basado en Playwright. Ahora sólo fuerza rediscover automático.
    parser.add_argument("--discover", action="store_true",
                        help="[deprecated] Alias de --force-rediscover. "
                             "El discovery ahora es 100% automático y se ejecuta "
                             "on-demand con TTL=7d sin Playwright.")
    args = parser.parse_args()

    if args.discover:
        log.info("--discover es alias deprecated de --force-rediscover.")
        args.force_rediscover = True

    if args.list_games:
        list_games()
        return

    markets = (("total_over_under", "team_with_more")
               if args.markets == "all" else (args.markets,))
    fetch_odds(
        dry_run=args.dry_run,
        markets=markets,
        max_matches=args.max_matches,
        force_rediscover=args.force_rediscover,
    )


if __name__ == "__main__":
    main()
