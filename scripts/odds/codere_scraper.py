"""
Codere Scraper — Throw-ins odds
================================
Scraper completamente autónomo de las cuotas de saques de banda (over/under) de
Codere para los partidos de LaLiga. Dos modos:

1. DISCOVERY (--discover): abre Playwright, navega a la página de fútbol de
   Codere, intercepta todas las llamadas XHR/Fetch, identifica los endpoints
   que devuelven (a) listado de eventos de LaLiga y (b) mercados por evento.
   Guarda los patrones en `data/reference/codere_endpoints.json`.

2. FETCH (default): usa los endpoints descubiertos para scrapear cuotas vía
   `requests` directamente, sin navegador. Mucho más rápido y estable.
   Itera todos los partidos activos de LaLiga, busca el mercado
   "saques de banda" (over/under) y lo guarda en
   `data/reference/odds_history.parquet` con una fila por (match, timestamp, line).

Uso:
  python scripts/odds/codere_scraper.py --discover             # 1 vez, cuando el mercado esté abierto
  python scripts/odds/codere_scraper.py                        # fetch, puede lanzarse cada N min
  python scripts/odds/codere_scraper.py --dry-run              # fetch sin guardar (debug)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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
        logging.FileHandler("codere_scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("codere_scraper")

CONFIG = {
    "codere_laliga_url": "https://www.codere.es/deportes/#/HorseLandingPage/Fútbol/España/LaLiga",
    "endpoints_path": "data/reference/codere_endpoints.json",
    "odds_history_path": "data/reference/odds_history.parquet",
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # Patrones de detección de mercado "saques de banda".
    # En DISCOVERY imprimimos todo y ajustamos si es necesario.
    "throw_in_keywords": [
        "saque", "saques de banda", "throw-in", "throw in", "throwin",
    ],
}

# ─────────────────────────────────────────────────────────────
# DISCOVERY (Playwright)
# ─────────────────────────────────────────────────────────────

def discover() -> None:
    """Abre Codere con Playwright, navega a LaLiga, intercepta XHR/Fetch y
    guarda los endpoints que devuelven JSON con partidos/mercados.

    Requiere:
      pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error("Playwright no instalado. Ejecuta: pip install playwright && playwright install chromium")
        sys.exit(2)

    captured: list[dict] = []

    log.info("Iniciando discovery de Codere LaLiga ...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # visible para depurar
        context = browser.new_context(user_agent=CONFIG["user_agent"])
        page = context.new_page()

        def on_response(response):
            try:
                url = response.url
                content_type = response.headers.get("content-type", "")
                if "json" not in content_type.lower():
                    return
                # Filtrar solo llamadas de la API (no estáticos)
                if not any(k in url.lower() for k in ("codere", "api", "sports", "event", "market", "fixture")):
                    return
                body_text = ""
                try:
                    body_text = response.text()[:2000]
                except Exception:
                    return
                captured.append({
                    "url": url,
                    "status": response.status,
                    "method": response.request.method,
                    "content_type": content_type,
                    "body_preview": body_text,
                })
            except Exception as exc:
                log.debug("on_response error: %s", exc)

        page.on("response", on_response)

        log.info("Navegando a %s ...", CONFIG["codere_laliga_url"])
        try:
            page.goto(CONFIG["codere_laliga_url"], wait_until="domcontentloaded", timeout=60_000)
        except Exception as exc:
            log.warning("Timeout inicial (%s). Continuando con lo capturado hasta ahora.", exc)

        log.info("Esperando 25s para que la SPA cargue y dispare XHRs ...")
        page.wait_for_timeout(25_000)

        log.info("Para capturar mercado de saques, abre MANUALMENTE un partido en la ventana y su mercado de saques de banda.")
        log.info("Tienes 60 segundos, pulsa Ctrl+C en consola para salir antes si ya ves las llamadas.")
        try:
            page.wait_for_timeout(60_000)
        except KeyboardInterrupt:
            pass

        browser.close()

    out_path = Path(CONFIG["endpoints_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"captured_at": datetime.now(timezone.utc).isoformat(),
                   "requests": captured}, f, indent=2, ensure_ascii=False)
    log.info("Guardadas %d llamadas JSON en %s", len(captured), out_path)

    # Imprimir resumen para que el usuario identifique los endpoints
    print()
    print("=" * 80)
    print("RESUMEN DE ENDPOINTS CAPTURADOS (para identificar LaLiga y saques de banda)")
    print("=" * 80)
    unique_urls = {}
    for c in captured:
        # normalizar URL para agrupar (quitar parámetros numéricos)
        norm = re.sub(r"/\d+", "/{id}", c["url"].split("?")[0])
        unique_urls.setdefault(norm, []).append(c["url"])
    for norm, urls in sorted(unique_urls.items()):
        sample = urls[0]
        preview = next((c["body_preview"][:150] for c in captured if c["url"] == sample), "")
        print(f"\n[{len(urls):3d}x] {norm}")
        print(f"      ejemplo: {sample}")
        print(f"      body:    {preview}...")

    print()
    print("SIGUIENTE PASO:")
    print("  1. Identifica en la lista cuál es el endpoint que devuelve partidos de LaLiga")
    print("  2. Identifica cuál devuelve mercados con 'saques de banda'")
    print(f"  3. Edita manualmente {out_path} añadiendo campos:")
    print('       "events_endpoint": "URL o patrón con {liga_id}",')
    print('       "markets_endpoint": "URL o patrón con {event_id}"')
    print(f"  4. Ejecuta: python scripts/odds/codere_scraper.py (modo fetch)")


# ─────────────────────────────────────────────────────────────
# FETCH (requests, usa endpoints descubiertos)
# ─────────────────────────────────────────────────────────────

def _load_endpoints() -> dict:
    p = Path(CONFIG["endpoints_path"])
    if not p.exists():
        log.error("No hay endpoints guardados en %s — ejecuta primero con --discover.", p)
        sys.exit(2)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "events_endpoint" not in data or "markets_endpoint" not in data:
        log.error(
            "El fichero %s no tiene 'events_endpoint' y 'markets_endpoint'. "
            "Edítalo manualmente tras el discovery.", p,
        )
        sys.exit(2)
    return data


def _match_throw_in_market(market_name: str) -> bool:
    name_lower = (market_name or "").lower()
    return any(k in name_lower for k in CONFIG["throw_in_keywords"])


def fetch_odds(dry_run: bool = False) -> pd.DataFrame:
    endpoints = _load_endpoints()
    sess = requests.Session()
    sess.headers.update({"User-Agent": CONFIG["user_agent"]})

    log.info("Descargando listado de eventos de LaLiga ...")
    events_url = endpoints["events_endpoint"]
    r = sess.get(events_url, timeout=30)
    r.raise_for_status()
    events_json = r.json()

    # Extracción del listado de eventos. Como no conocemos el schema exacto hasta
    # el discovery, lo hacemos genérico: buscamos listas con objetos que tengan
    # campos tipo "id", "name", "startDate" / "date" / "event_id".
    events = _extract_events(events_json)
    log.info("Eventos de LaLiga encontrados: %d", len(events))
    if not events:
        log.warning("No se extrajeron eventos — revisa _extract_events() y el schema devuelto.")
        return pd.DataFrame()

    rows = []
    now = datetime.now(timezone.utc)
    for ev in events:
        event_id = ev.get("id") or ev.get("event_id")
        if not event_id:
            continue
        markets_url = endpoints["markets_endpoint"].format(event_id=event_id)
        try:
            r = sess.get(markets_url, timeout=30)
            r.raise_for_status()
            markets_json = r.json()
        except Exception as exc:
            log.warning("Evento %s (%s vs %s): %s", event_id, ev.get("home"), ev.get("away"), exc)
            continue

        markets = _extract_markets(markets_json)
        for mkt in markets:
            if not _match_throw_in_market(mkt.get("name", "")):
                continue
            for sel in mkt.get("selections", []):
                line = sel.get("line")
                side = sel.get("side")    # "over" | "under"
                price = sel.get("price")
                if line is None or side is None or price is None:
                    continue
                rows.append({
                    "match_id": int(event_id) if str(event_id).isdigit() else str(event_id),
                    "home_team": ev.get("home"),
                    "away_team": ev.get("away"),
                    "match_date": ev.get("start"),
                    "scraped_at": now,
                    "hours_before": _hours_before(now, ev.get("start")),
                    "market_name": mkt.get("name"),
                    "line": float(line),
                    "side": side,
                    "odds": float(price),
                    "bookmaker": "codere",
                })
        time.sleep(0.2)  # rate limiting suave

    df = pd.DataFrame(rows)
    log.info("Cuotas extraídas: %d filas para %d partidos", len(df), df["match_id"].nunique() if len(df) else 0)

    if dry_run:
        print(df.head(30).to_string(index=False))
        return df

    if len(df):
        _append_to_history(df)
    return df


def _extract_events(events_json) -> list[dict]:
    """Normaliza el JSON de eventos a lista de dicts con campos id, home, away, start.

    Placeholder flexible — ajustar según schema real tras discovery.
    """
    events = []
    # Búsqueda genérica de listas de eventos
    candidates = []
    if isinstance(events_json, list):
        candidates = events_json
    elif isinstance(events_json, dict):
        # busca cualquier clave que contenga "event" o "fixture" o "match"
        for k, v in events_json.items():
            if isinstance(v, list) and len(v) and isinstance(v[0], dict):
                if any(kw in k.lower() for kw in ("event", "fixture", "match", "item")):
                    candidates = v
                    break
        if not candidates:
            # fallback: la primera lista de dicts del JSON
            for v in events_json.values():
                if isinstance(v, list) and len(v) and isinstance(v[0], dict):
                    candidates = v
                    break

    for c in candidates:
        # Normalización flexible de campos
        ev_id = c.get("id") or c.get("eventId") or c.get("event_id")
        home = c.get("home") or c.get("homeTeam") or c.get("home_team") or c.get("participant1")
        away = c.get("away") or c.get("awayTeam") or c.get("away_team") or c.get("participant2")
        start = c.get("startDate") or c.get("startTime") or c.get("start") or c.get("date")
        if isinstance(home, dict):
            home = home.get("name")
        if isinstance(away, dict):
            away = away.get("name")
        if ev_id and home and away:
            events.append({"id": ev_id, "home": home, "away": away, "start": start})
    return events


def _extract_markets(markets_json) -> list[dict]:
    """Normaliza el JSON de mercados a lista [{name, selections: [{line, side, price}]}].
    Placeholder flexible.
    """
    markets_raw = []
    if isinstance(markets_json, list):
        markets_raw = markets_json
    elif isinstance(markets_json, dict):
        for k, v in markets_json.items():
            if isinstance(v, list) and len(v) and isinstance(v[0], dict):
                if "market" in k.lower() or "bet" in k.lower():
                    markets_raw = v
                    break
        if not markets_raw:
            for v in markets_json.values():
                if isinstance(v, list) and len(v) and isinstance(v[0], dict):
                    markets_raw = v
                    break

    out = []
    for m in markets_raw:
        name = m.get("name") or m.get("marketName") or m.get("title") or ""
        sels_raw = m.get("selections") or m.get("outcomes") or m.get("runners") or []
        sels = []
        for s in sels_raw:
            line = s.get("line") or s.get("handicap") or s.get("points")
            side_raw = (s.get("name") or s.get("label") or s.get("side") or "").lower()
            side = "over" if "over" in side_raw or "más" in side_raw or "mas" in side_raw else (
                   "under" if "under" in side_raw or "menos" in side_raw else None)
            price = s.get("price") or s.get("odds") or s.get("decimal") or s.get("decimalOdds")
            if line is not None and side is not None and price is not None:
                sels.append({"line": line, "side": side, "price": price})
        if sels:
            out.append({"name": name, "selections": sels})
    return out


def _hours_before(now: datetime, start_iso: str | None) -> float | None:
    if not start_iso:
        return None
    try:
        start = pd.to_datetime(start_iso, utc=True).to_pydatetime()
    except Exception:
        return None
    return (start - now).total_seconds() / 3600.0


def _append_to_history(df: pd.DataFrame) -> None:
    out = Path(CONFIG["odds_history_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = pd.read_parquet(out)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(out, index=False)
    log.info("odds_history actualizado: %s (%d filas totales)", out, len(df))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover", action="store_true",
                        help="Modo discovery: abre Playwright e intercepta endpoints.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch sin guardar a parquet.")
    args = parser.parse_args()

    if args.discover:
        discover()
    else:
        fetch_odds(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
