"""
WhoScored LaLiga Scraper — versión completa con MongoDB
========================================================
Extrae de es.whoscored.com para cada partido de LaLiga:

  1. throw_ins       → Dataset principal para el modelo predictivo
  2. pass_map        → Todos los pases con origen+destino (x,y → endX,endY)
  3. heatmap         → Todos los toques con coordenadas por jugador y equipo
  4. all_events      → Todos los eventos raw del partido
  5. team_stats      → Estadísticas agregadas de equipo por partido
  6. raw/{id}.json   → JSON crudo por partido (caché local)

Cada output se guarda en:
  - MongoDB Atlas (en tiempo real, por partido)
  - CSV/Parquet local (consolidado al finalizar)
"""
from __future__ import annotations

import json
import time
import random
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeout

from dotenv import load_dotenv
import os
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, OperationFailure

load_dotenv()


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

CONFIG = {
    "region_id":     206,
    "tournament_id": 4,

    "known_seasons": {
        "2025/2026": {"season_id": 10803, "stage_id": 24622},
        "2024/2025": {"season_id": 10247, "stage_id": 23815},
        "2023/2024": {"season_id":  9715, "stage_id": 23277},
        "2022/2023": {"season_id":  9098, "stage_id": 22580},
        "2021/2022": {"season_id":  8558, "stage_id": 21963},
        "2020/2021": {"season_id":  8016, "stage_id": 21413},
        "2019/2020": {"season_id":  7466, "stage_id": 20486},
    },

    "output_dir":    "data/whoscored_laliga",
    "output_format": "both",   # "csv", "parquet" o "both"

    "headless":      True,
    "delay_min":     4.0,
    "delay_max":     9.0,
    "max_retries":   3,
    "timeout_ms":    35_000,
}


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

def setup_logging(log_file: str = "scraper.log"):
    """Configura el root logger. Debe llamarse desde el punto de entrada,
    nunca al importar el módulo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# QUALIFIER IDs DE WHOSCORED
# ─────────────────────────────────────────────────────────────

Q = {
    "THROW_IN":        107,
    "PASS_END_X":      140,
    "PASS_END_Y":      141,
    "ZONE":             56,
    "LENGTH":          212,
    "ANGLE":           213,
    "LONG_BALL":         1,
    "CROSS":             2,
    "HEAD_PASS":         3,
    "THROUGH_BALL":      4,
    "FREEKICK":          5,
    "CORNER":            6,
    "LEFT_FOOT":        72,
    "RIGHT_FOOT":       20,
    "KEY_PASS":      11113,
    "INTENT_ASSIST":   154,
    "GOAL_ASSIST":   11111,
    "SHOT_ASSIST":     210,
    "BIG_CHANCE":    11112,
    "CHIPPED":         155,
    "LAYOFF":          156,
    "FIRST_TOUCH":     328,
    "OFFENSIVE":       286,
    "DEFENSIVE":       285,
    "GOAL_KICK":       124,
    "KEEPER_THROW":    123,
    "BLOCKED_X":       146,
    "BLOCKED_Y":       147,
    "GOAL_MOUTH_Y":    102,
    "GOAL_MOUTH_Z":    103,
    "STANDING_SAVE":   178,
    "HANDS":           182,
    "LAST_MAN":         14,
    "XG":              321,
}


# ─────────────────────────────────────────────────────────────
# MONGODB
# ─────────────────────────────────────────────────────────────

class MongoManager:
    """Gestiona la conexión y operaciones con MongoDB Atlas."""

    COLLECTIONS = {
        "throw_ins":   ("match_id", "event_id"),
        "pass_map":    ("match_id", "event_id"),
        "heatmap":     ("match_id", "event_id"),
        "all_events":  ("match_id", "event_id"),
        "team_stats":  ("match_id", "team_id"),
        "raw_matches": ("match_id",),
    }

    def __init__(self):
        uri     = os.getenv("MONGO_URI")
        db_name = os.getenv("MONGO_DB", "modelthrowins")
        if not uri:
            raise ValueError("MONGO_URI no encontrada en .env")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
        self.client.admin.command("ping")
        self.db = self.client[db_name]

        # Crear referencias y índices
        self.throw_ins   = self.db["throw_ins"]
        self.pass_map    = self.db["pass_map"]
        self.heatmap     = self.db["heatmap"]
        self.all_events  = self.db["all_events"]
        self.team_stats  = self.db["team_stats"]
        self.raw_matches = self.db["raw_matches"]

        self.throw_ins.create_index  ([("match_id", 1), ("event_id",  1)], unique=True)
        self.pass_map.create_index   ([("match_id", 1), ("event_id",  1)], unique=True)
        self.heatmap.create_index    ([("match_id", 1), ("event_id",  1)], unique=True)
        self.all_events.create_index ([("match_id", 1), ("event_id",  1)], unique=True)
        self.team_stats.create_index ([("match_id", 1), ("team_id",   1)], unique=True)
        self.raw_matches.create_index("match_id", unique=True)

        log.info(f"Conectado a MongoDB Atlas — base de datos: {db_name}")

    def _upsert_df(self, collection, df: pd.DataFrame,
                   key1: str, key2: str | None = None) -> int:
        """Upsert genérico de un DataFrame en una colección."""
        if df.empty:
            return 0
        ops = []
        for row in df.to_dict("records"):
            filter_q = {key1: row[key1]}
            if key2:
                filter_q[key2] = row[key2]
            ops.append(UpdateOne(filter_q, {"$set": row}, upsert=True))
        try:
            result = collection.bulk_write(ops, ordered=False)
            return result.upserted_count + result.modified_count
        except BulkWriteError as e:
            log.warning(f"BulkWriteError (duplicados ignorados): {e.details.get('nInserted', 0)} insertados")
            return 0

    def save_throw_ins(self, df: pd.DataFrame) -> int:
        return self._upsert_df(self.throw_ins, df, "match_id", "event_id")

    def save_pass_map(self, df: pd.DataFrame) -> int:
        return self._upsert_df(self.pass_map, df, "match_id", "event_id")

    def save_heatmap(self, df: pd.DataFrame) -> int:
        return self._upsert_df(self.heatmap, df, "match_id", "event_id")

    def save_all_events(self, df: pd.DataFrame) -> int:
        return self._upsert_df(self.all_events, df, "match_id", "event_id")

    def save_team_stats(self, df: pd.DataFrame) -> int:
        return self._upsert_df(self.team_stats, df, "match_id", "team_id")

    def save_raw(self, match_id: int, raw: dict):
        self.raw_matches.update_one(
            {"match_id": match_id},
            {"$set": {"match_id": match_id, "data": raw,
                      "saved_at": datetime.utcnow()}},
            upsert=True,
        )

    def get_processed_ids(self, season: str = None) -> set:
        """IDs ya guardados en team_stats para una temporada concreta."""
        query = {"season": season} if season else {}
        return set(self.team_stats.distinct("match_id", query))

    def summary(self) -> dict:
        return {
            "throw_ins":   self.throw_ins.count_documents({}),
            "pass_map":    self.pass_map.count_documents({}),
            "heatmap":     self.heatmap.count_documents({}),
            "all_events":  self.all_events.count_documents({}),
            "team_stats":  self.team_stats.count_documents({}),
            "raw_matches": self.raw_matches.count_documents({}),
        }

    def close(self):
        self.client.close()


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def random_delay(lo: float = None, hi: float = None):
    time.sleep(random.uniform(lo or CONFIG["delay_min"],
                              hi or CONFIG["delay_max"]))


def qval(qualifiers: list, q_id: int) -> Optional[str]:
    for q in qualifiers:
        if q.get("type", {}).get("value") == q_id:
            return q.get("value")
    return None


def qhas(qualifiers: list, q_id: int) -> bool:
    return any(q.get("type", {}).get("value") == q_id for q in qualifiers)


def qfloat(qualifiers: list, q_id: int, default: float = 0.0) -> float:
    v = qval(qualifiers, q_id)
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN DE MATCH IDs
# ─────────────────────────────────────────────────────────────

def get_match_ids(page: Page, season_id: int, stage_id: int) -> list[int]:
    """
    Extrae TODOS los match IDs de una temporada de LaLiga.

    Estrategia dual:
    1. Temporada actual: /tournaments/{stageId}/data/?d=YYYYMM (10 llamadas)
    2. Cualquier temporada: /livescores/data/?d=YYYYMMDD iterando día a día
       filtrando por regionId=206, tournamentId=4 (LaLiga España)
    """
    ids: set[int] = set()
    season_months = _get_season_months(season_id)
    y1 = int(season_months[0][:4])
    log.info(f"Extrayendo match IDs (season={season_id}, stage={stage_id})")

    # Cargar cualquier página de WhoScored para tener cookies de sesión
    try:
        page.goto("https://es.whoscored.com/livescores",
                   wait_until="networkidle", timeout=CONFIG["timeout_ms"])
        random_delay(2, 3)
    except PlaywrightTimeout:
        log.warning("  Timeout cargando livescores, continuando...")

    # Método 1: API de fixtures por mes (solo funciona para temporada actual)
    api_ids = _get_ids_via_tournament_api(page, stage_id, season_months)
    if api_ids:
        ids.update(api_ids)
        log.info(f"  API tournament: {len(api_ids)} IDs")

    # Método 2: API de livescores día a día (funciona para TODAS las temporadas)
    if len(ids) < 300:
        log.info("  Usando livescores API dia a dia...")
        ls_ids = _get_ids_via_livescores(page, season_months)
        new = len(ls_ids - ids)
        ids.update(ls_ids)
        log.info(f"  Livescores API: +{new} IDs nuevos (total {len(ids)})")

    log.info(f"Total match IDs encontrados: {len(ids)}")
    return sorted(ids)


# Filtro JS reutilizable: solo Primera División española
_JS_LALIGA_FILTER = """
function isLaLigaPrimera(t) {
    if (t.regionId !== 206 || t.tournamentId !== 4) return false;
    const name = (t.stageName || t.tournamentName || '').toLowerCase();
    if (name.includes('2') || name.includes('hypermotion') ||
        name.includes('smartbank') || name.includes('adelante') ||
        name.includes('copa') || name.includes('super')) return false;
    return true;
}
"""


def _get_ids_via_tournament_api(page: Page, stage_id: int,
                                 season_months: list) -> set:
    """Intenta obtener IDs usando /tournaments/{stageId}/data/?d=YYYYMM."""
    ids = set()
    for date_str in season_months:
        try:
            result = page.evaluate(f"""
                async () => {{
                    try {{
                        {_JS_LALIGA_FILTER}
                        const r = await fetch('/tournaments/{stage_id}/data/?d={date_str}&isAggregate=false', {{
                            headers: {{ 'X-Requested-With': 'XMLHttpRequest' }},
                            credentials: 'include'
                        }});
                        if (!r.ok) return null;
                        const j = await r.json();
                        const ids = [];
                        for (const t of (j.tournaments || [])) {{
                            if (isLaLigaPrimera(t)) {{
                                for (const m of (t.matches || [])) {{
                                    if (m.id && m.status === 6) ids.push(m.id);
                                }}
                            }}
                        }}
                        return ids;
                    }} catch(e) {{ return null; }}
                }}
            """)
            if result:
                ids.update(result)
            random_delay(0.5, 1)
        except Exception:
            pass
    return ids


def _get_ids_via_livescores(page: Page, season_months: list) -> set:
    """
    Obtiene IDs iterando día a día con /livescores/data/?d=YYYYMMDD.
    Filtra estrictamente por LaLiga Primera División:
      - regionId=206 (España)
      - tournamentId=4 (LaLiga)
      - stageName exacto "LaLiga" (excluye "LaLiga 2", Hypermotion, etc.)
      - status=6 (partido finalizado)
    Procesa un mes completo por evaluate() para minimizar overhead.
    """
    all_ids: set[int] = set()

    for month_str in season_months:
        year = int(month_str[:4])
        month = int(month_str[4:])
        try:
            result = page.evaluate(f"""
                async () => {{
                    {_JS_LALIGA_FILTER}
                    const year = {year};
                    const month = {month};
                    const daysInMonth = new Date(year, month, 0).getDate();
                    const ids = [];
                    for (let day = 1; day <= daysInMonth; day++) {{
                        const d = String(year)
                            + String(month).padStart(2, '0')
                            + String(day).padStart(2, '0');
                        try {{
                            const r = await fetch('/livescores/data/?d=' + d, {{
                                headers: {{ 'X-Requested-With': 'XMLHttpRequest' }},
                                credentials: 'include'
                            }});
                            const j = await r.json();
                            for (const t of (j.tournaments || [])) {{
                                if (isLaLigaPrimera(t)) {{
                                    for (const m of (t.matches || [])) {{
                                        if (m.id && m.status === 6) ids.push(m.id);
                                    }}
                                }}
                            }}
                        }} catch(e) {{}}
                    }}
                    return ids;
                }}
            """)
            if result:
                before = len(all_ids)
                all_ids.update(result)
                log.info(f"  {month_str}: +{len(all_ids) - before} IDs (total {len(all_ids)})")
        except Exception as e:
            log.warning(f"  {month_str}: error livescores: {e}")

    return all_ids


def _get_season_months(season_id: int) -> list[str]:
    """
    Devuelve los meses (formato YYYYMM) de una temporada de LaLiga
    basándose en el season_id conocido.
    """
    # Derivar el año de inicio desde CONFIG — fuente de verdad única
    y1 = 2024  # fallback: solo alcanzable con season_id desconocido
    for key, cfg in CONFIG["known_seasons"].items():
        if cfg["season_id"] == season_id:
            y1 = int(key[:4])
            break
    y2 = y1 + 1

    # Temporada: agosto y1 → junio y2
    months = []
    for m in range(8, 13):   # ago, sep, oct, nov, dic del año 1
        months.append(f"{y1}{m:02d}")
    for m in range(1, 7):    # ene, feb, mar, abr, may, jun del año 2
        months.append(f"{y2}{m:02d}")
    return months


# ─────────────────────────────────────────────────────────────
# EXTRACCIÓN DE matchCentreData
# ─────────────────────────────────────────────────────────────

def fetch_match_data(page: Page, match_id: int) -> Optional[dict]:
    url = f"https://es.whoscored.com/matches/{match_id}/live"
    for attempt in range(1, CONFIG["max_retries"] + 1):
        try:
            log.info(f"  Partido {match_id} (intento {attempt})...")
            page.goto(url, wait_until="networkidle", timeout=CONFIG["timeout_ms"])
            random_delay(1.5, 3.0)

            data = page.evaluate("""
                () => {
                    try {
                        const args = require.config.params["args"];
                        if (!args || !args.matchCentreData) return null;
                        return args;
                    } catch(e) {
                        return { _error: e.toString() };
                    }
                }
            """)

            if data and "_error" not in data and data.get("matchCentreData"):
                n_events = len(data["matchCentreData"].get("events", []))
                log.info(f"  ✓ {match_id}: {n_events} eventos")
                return data

            log.warning(f"  Sin datos para {match_id}: {str(data)[:100]}")

        except PlaywrightTimeout:
            log.warning(f"  Timeout partido {match_id} (intento {attempt})")
        except Exception as e:
            log.error(f"  Error partido {match_id}: {e}")

        if attempt < CONFIG["max_retries"]:
            random_delay(15, 25)
    return None


# ─────────────────────────────────────────────────────────────
# ESTRUCTURAS AUXILIARES
# ─────────────────────────────────────────────────────────────

def _player_dict(mcd: dict) -> dict:
    d = {}
    for tk in ["home", "away"]:
        for p in mcd[tk].get("players", []):
            d[p["playerId"]] = {
                "name":       p.get("name", ""),
                "position":   p.get("position", ""),
                "height":     p.get("height", 0),
                "weight":     p.get("weight", 0),
                "age":        p.get("age", 0),
                "jersey":     p.get("shirtNo", 0),
                "is_starter": p.get("isFirstEleven", False),
                "team_side":  tk,
                "team_id":    mcd[tk]["teamId"],
            }
    return d


def _formation_lookup(mcd: dict) -> dict:
    lk = {}
    for tk in ["home", "away"]:
        tid = mcd[tk]["teamId"]
        for f in mcd[tk].get("formations", []):
            s = int(f.get("startMinuteExpanded", 0))
            e = int(f.get("endMinuteExpanded", 200))
            name = f.get("formationName", "")
            for m in range(s, e + 1):
                lk[(tid, m)] = name
    return lk


def _score_timeline(events: list, home_id: int, away_id: int) -> dict:
    tl = {0: (0, 0)}
    hg = ag = 0
    goals = sorted(
        [e for e in events if e.get("type", {}).get("displayName") == "Goal"],
        key=lambda e: e.get("expandedMinute", e.get("minute", 0)),
    )
    prev_m = 0
    for g in goals:
        gm = g.get("expandedMinute", g.get("minute", 0))
        for m in range(prev_m, gm):
            tl[m] = (hg, ag)
        if g.get("teamId") == home_id:
            hg += 1
        else:
            ag += 1
        tl[gm] = (hg, ag)
        prev_m = gm
    for m in range(prev_m, 130):
        tl[m] = (hg, ag)
    return tl


def _score_at(tl: dict, minute: int) -> tuple[int, int]:
    if minute in tl:
        return tl[minute]
    candidates = [k for k in tl if k <= minute]
    return tl[max(candidates)] if candidates else (0, 0)


def _recent_ctx(events: list, current_idx: int, team_id: int,
                window: int = 5) -> dict:
    cur_min = events[current_idx].get("expandedMinute",
              events[current_idx].get("minute", 0))
    lo = cur_min - window

    # Scan inverso con early-exit: los eventos están ordenados cronológicamente,
    # así que en cuanto el minuto sea < lo podemos parar.
    team_ev: list = []
    all_ev:  list = []
    for e in reversed(events[:current_idx]):
        em = e.get("expandedMinute", e.get("minute", 0))
        if em < lo:
            break
        all_ev.append(e)
        if e.get("teamId") == team_id:
            team_ev.append(e)

    passes      = [e for e in team_ev if e.get("type", {}).get("displayName") == "Pass"]
    pass_ok     = sum(1 for p in passes
                      if p.get("outcomeType", {}).get("displayName") == "Successful")
    throw_ins   = sum(1 for p in passes if qhas(p.get("qualifiers", []), Q["THROW_IN"]))
    t_touches   = sum(1 for e in team_ev if e.get("isTouch"))
    all_touches = sum(1 for e in all_ev  if e.get("isTouch"))
    tackles_won = sum(1 for e in team_ev
                      if e.get("type", {}).get("displayName") == "Tackle"
                      and e.get("outcomeType", {}).get("displayName") == "Successful")
    intercepts  = sum(1 for e in team_ev
                      if e.get("type", {}).get("displayName") == "Interception")
    pressures   = sum(1 for e in all_ev
                      if e.get("teamId") != team_id
                      and e.get("type", {}).get("displayName") in
                      ["Tackle", "Challenge", "Foul"])
    return {
        "passes_completed":    pass_ok,
        "passes_attempted":    len(passes),
        "pass_accuracy":       pass_ok / len(passes) if passes else 0.0,
        "throw_ins_recent":    throw_ins,
        "possession_pct":      t_touches / all_touches if all_touches else 0.0,
        "tackles_won_recent":  tackles_won,
        "interceptions_recent":intercepts,
        "pressures_received":  pressures,
    }


# ─────────────────────────────────────────────────────────────
# PROCESAMIENTO — 4 OUTPUTS + TEAM STATS
# ─────────────────────────────────────────────────────────────

def process_match(raw: dict, season: str) -> dict[str, pd.DataFrame]:
    """
    Convierte matchCentreData en 4 DataFrames:
      throw_ins  / pass_map / heatmap / all_events
    """
    mcd      = raw["matchCentreData"]
    events   = mcd.get("events", [])
    match_id = raw["matchId"]

    empty = {k: pd.DataFrame() for k in
             ["throw_ins", "pass_map", "heatmap", "all_events"]}
    if not events:
        return empty

    meta = {
        "match_id":    match_id,
        "season":      season,
        "home_team":   mcd["home"]["name"],
        "away_team":   mcd["away"]["name"],
        "home_team_id":mcd["home"]["teamId"],
        "away_team_id":mcd["away"]["teamId"],
        "score":       mcd.get("score", ""),
        "ht_score":    mcd.get("htScore", ""),
        "ft_score":    mcd.get("ftScore", ""),
        "venue":       mcd.get("venueName", ""),
        "attendance":  mcd.get("attendance", 0),
        "referee":     mcd.get("referee", {}).get("name", ""),
        "match_date":  mcd.get("startDate", ""),
        "match_time":  mcd.get("startTime", ""),
    }

    players    = _player_dict(mcd)
    formations = _formation_lookup(mcd)
    score_tl   = _score_timeline(events,
                                 mcd["home"]["teamId"],
                                 mcd["away"]["teamId"])
    return {
        "throw_ins":  _build_throw_ins(events, meta, players, formations, score_tl, mcd),
        "pass_map":   _build_pass_map(events, meta, players),
        "heatmap":    _build_heatmap(events, meta, players),
        "all_events": _build_all_events(events, meta, players),
    }


# ── 1. THROW-INS ─────────────────────────────────────────────

def _build_throw_ins(events, meta, players, formations,
                     score_tl, mcd) -> pd.DataFrame:
    rows = []
    for idx, ev in enumerate(events):
        if ev.get("type", {}).get("displayName") != "Pass":
            continue
        qs = ev.get("qualifiers", [])
        if not qhas(qs, Q["THROW_IN"]):
            continue

        minute    = ev.get("expandedMinute", ev.get("minute", 0))
        team_id   = ev.get("teamId")
        player_id = ev.get("playerId")
        p         = players.get(player_id, {})
        is_home   = (team_id == meta["home_team_id"])

        x     = ev.get("x", 0.0)
        y     = ev.get("y", 0.0)
        end_x = ev.get("endX") or qfloat(qs, Q["PASS_END_X"])
        end_y = ev.get("endY") or qfloat(qs, Q["PASS_END_Y"])

        hg, ag   = _score_at(score_tl, minute)
        my_goals = hg if is_home else ag
        rv_goals = ag if is_home else hg
        diff     = my_goals - rv_goals

        field_zone = ("defensive_third" if x < 34 else
                      "middle_third"    if x < 67 else
                      "attacking_third")
        side = "left" if y < 50 else "right"
        ctx  = _recent_ctx(events, idx, team_id, window=5)

        nxt  = events[idx + 1] if idx + 1 < len(events) else None
        nxt2 = events[idx + 2] if idx + 2 < len(events) else None
        nxt3 = events[idx + 3] if idx + 3 < len(events) else None

        retained  = bool(nxt  and nxt.get("teamId")  == team_id)
        retained3 = all(e and e.get("teamId") == team_id for e in [nxt, nxt2, nxt3])

        rows.append({
            "match_id":               meta["match_id"],
            "season":                 meta["season"],
            "event_id":               ev.get("id"),
            "event_idx":              idx,
            "home_team":              meta["home_team"],
            "away_team":              meta["away_team"],
            "venue":                  meta["venue"],
            "attendance":             meta["attendance"],
            "match_date":             meta["match_date"],
            "minute":                 ev.get("minute", 0),
            "second":                 ev.get("second", 0),
            "expanded_minute":        minute,
            "period":                 ev.get("period", {}).get("displayName", ""),
            "period_value":           ev.get("period", {}).get("value", 0),
            "team_id":                team_id,
            "is_home_team":           int(is_home),
            "player_id":              player_id,
            "player_name":            p.get("name", ""),
            "player_position":        p.get("position", ""),
            "player_age":             p.get("age", 0),
            "player_height":          p.get("height", 0),
            "player_is_starter":      int(p.get("is_starter", False)),
            "x":                      x,
            "y":                      y,
            "end_x":                  end_x,
            "end_y":                  end_y,
            "pass_length":            qfloat(qs, Q["LENGTH"]),
            "pass_angle":             qfloat(qs, Q["ANGLE"]),
            "x_gain":                 end_x - x,
            "y_gain":                 abs(end_y - y),
            "dist_to_goal_origin":    ((100 - x)**2  + (50 - y)**2)   ** 0.5,
            "dist_to_goal_dest":      ((100 - end_x)**2 + (50 - end_y)**2) ** 0.5,
            "field_zone":             field_zone,
            "field_side":             side,
            "near_corner_flag":       int(x > 85 or x < 15),
            "zone_raw":               qval(qs, Q["ZONE"]) or "",
            "is_long_ball":           int(qhas(qs, Q["LONG_BALL"])),
            "is_chipped":             int(qhas(qs, Q["CHIPPED"])),
            "is_head_pass":           int(qhas(qs, Q["HEAD_PASS"])),
            "is_layoff":              int(qhas(qs, Q["LAYOFF"])),
            "is_first_touch":         int(qhas(qs, Q["FIRST_TOUCH"])),
            "is_offensive":           int(qhas(qs, Q["OFFENSIVE"])),
            "outcome":                ev.get("outcomeType", {}).get("displayName", ""),
            "is_successful":          int(ev.get("outcomeType", {}).get("displayName") == "Successful"),
            "is_key_pass":            int(qhas(qs, Q["KEY_PASS"])),
            "is_assist":              int(qhas(qs, Q["INTENT_ASSIST"]) or qhas(qs, Q["GOAL_ASSIST"])),
            "is_shot_assist":         int(qhas(qs, Q["SHOT_ASSIST"])),
            "is_big_chance":          int(qhas(qs, Q["BIG_CHANCE"])),
            "retained_possession":    int(retained),
            "retained_possession_3":  int(retained3),
            "next_event_type":        nxt.get("type", {}).get("displayName", "") if nxt else "",
            "next_same_team":         int(nxt.get("teamId") == team_id) if nxt else 0,
            "my_goals":               my_goals,
            "rival_goals":            rv_goals,
            "score_diff":             diff,
            "is_winning":             int(diff > 0),
            "is_losing":              int(diff < 0),
            "is_drawing":             int(diff == 0),
            "team_formation":         formations.get((team_id, minute), ""),
            "ctx_passes_completed":   ctx["passes_completed"],
            "ctx_passes_attempted":   ctx["passes_attempted"],
            "ctx_pass_accuracy":      round(ctx["pass_accuracy"], 3),
            "ctx_throw_ins_recent":   ctx["throw_ins_recent"],
            "ctx_possession_pct":     round(ctx["possession_pct"], 3),
            "ctx_tackles_won":        ctx["tackles_won_recent"],
            "ctx_interceptions":      ctx["interceptions_recent"],
            "ctx_pressures_received": ctx["pressures_received"],
        })

    return pd.DataFrame(rows)


# ── 2. PASS MAP ───────────────────────────────────────────────

def _build_pass_map(events, meta, players) -> pd.DataFrame:
    """
    Todos los pases del partido con coordenadas origen y destino.
    Datos exactos que WhoScored usa para la Pizarra / Pass Map.
    """
    rows = []
    for ev in events:
        if ev.get("type", {}).get("displayName") != "Pass":
            continue
        qs        = ev.get("qualifiers", [])
        player_id = ev.get("playerId")
        p         = players.get(player_id, {})
        team_id   = ev.get("teamId")

        rows.append({
            "match_id":        meta["match_id"],
            "season":          meta["season"],
            "match_date":      meta["match_date"],
            "home_team":       meta["home_team"],
            "away_team":       meta["away_team"],
            "event_id":        ev.get("id"),
            "minute":          ev.get("minute", 0),
            "expanded_minute": ev.get("expandedMinute", ev.get("minute", 0)),
            "second":          ev.get("second", 0),
            "period":          ev.get("period", {}).get("displayName", ""),
            "team_id":         team_id,
            "is_home_team":    int(team_id == meta["home_team_id"]),
            "player_id":       player_id,
            "player_name":     p.get("name", ""),
            "player_position": p.get("position", ""),
            "x":               ev.get("x", 0.0),
            "y":               ev.get("y", 0.0),
            "end_x":           ev.get("endX") or qfloat(qs, Q["PASS_END_X"]),
            "end_y":           ev.get("endY") or qfloat(qs, Q["PASS_END_Y"]),
            "length":          qfloat(qs, Q["LENGTH"]),
            "angle":           qfloat(qs, Q["ANGLE"]),
            "zone":            qval(qs, Q["ZONE"]) or "",
            "outcome":         ev.get("outcomeType", {}).get("displayName", ""),
            "is_successful":   int(ev.get("outcomeType", {}).get("displayName") == "Successful"),
            "is_throw_in":     int(qhas(qs, Q["THROW_IN"])),
            "is_long_ball":    int(qhas(qs, Q["LONG_BALL"])),
            "is_cross":        int(qhas(qs, Q["CROSS"])),
            "is_head_pass":    int(qhas(qs, Q["HEAD_PASS"])),
            "is_through_ball": int(qhas(qs, Q["THROUGH_BALL"])),
            "is_freekick":     int(qhas(qs, Q["FREEKICK"])),
            "is_corner":       int(qhas(qs, Q["CORNER"])),
            "is_goal_kick":    int(qhas(qs, Q["GOAL_KICK"])),
            "is_key_pass":     int(qhas(qs, Q["KEY_PASS"])),
            "is_assist":       int(qhas(qs, Q["INTENT_ASSIST"]) or qhas(qs, Q["GOAL_ASSIST"])),
            "is_shot_assist":  int(qhas(qs, Q["SHOT_ASSIST"])),
            "is_chipped":      int(qhas(qs, Q["CHIPPED"])),
            "is_offensive":    int(qhas(qs, Q["OFFENSIVE"])),
            "is_left_foot":    int(qhas(qs, Q["LEFT_FOOT"])),
            "is_right_foot":   int(qhas(qs, Q["RIGHT_FOOT"])),
        })

    return pd.DataFrame(rows)


# ── 3. HEATMAP ────────────────────────────────────────────────

def _build_heatmap(events, meta, players) -> pd.DataFrame:
    """
    Todos los eventos con isTouch=True y coordenadas (x, y).
    Datos exactos que WhoScored usa para los Mapas de Actividad.
    Con este dataset: heatmaps con mplsoccer, centroide por jugador,
    densidad por zona, profundidad media de juego.
    """
    rows = []
    for ev in events:
        if not ev.get("isTouch"):
            continue
        x = ev.get("x", 0.0)
        y = ev.get("y", 0.0)
        if x == 0.0 and y == 0.0:
            continue

        player_id = ev.get("playerId")
        p         = players.get(player_id, {})
        team_id   = ev.get("teamId")

        rows.append({
            "match_id":        meta["match_id"],
            "season":          meta["season"],
            "match_date":      meta["match_date"],
            "home_team":       meta["home_team"],
            "away_team":       meta["away_team"],
            "event_id":        ev.get("id"),
            "minute":          ev.get("minute", 0),
            "expanded_minute": ev.get("expandedMinute", ev.get("minute", 0)),
            "second":          ev.get("second", 0),
            "period":          ev.get("period", {}).get("displayName", ""),
            "team_id":         team_id,
            "is_home_team":    int(team_id == meta["home_team_id"]),
            "player_id":       player_id,
            "player_name":     p.get("name", ""),
            "player_position": p.get("position", ""),
            "x":               x,
            "y":               y,
            "event_type":      ev.get("type", {}).get("displayName", ""),
            "outcome":         ev.get("outcomeType", {}).get("displayName", ""),
            "is_successful":   int(ev.get("outcomeType", {}).get("displayName") == "Successful"),
            "field_zone":      ("defensive_third" if x < 34 else
                                "middle_third"    if x < 67 else
                                "attacking_third"),
            "field_side":      "left" if y < 50 else "right",
        })

    return pd.DataFrame(rows)


# ── 4. ALL EVENTS ─────────────────────────────────────────────

def _build_all_events(events, meta, players) -> pd.DataFrame:
    rows = []
    for ev in events:
        qs        = ev.get("qualifiers", [])
        player_id = ev.get("playerId")
        p         = players.get(player_id, {})
        team_id   = ev.get("teamId")

        rows.append({
            "match_id":        meta["match_id"],
            "season":          meta["season"],
            "match_date":      meta["match_date"],
            "home_team":       meta["home_team"],
            "away_team":       meta["away_team"],
            "event_id":        ev.get("id"),
            "event_seq_id":    ev.get("eventId"),
            "minute":          ev.get("minute", 0),
            "second":          ev.get("second", 0),
            "expanded_minute": ev.get("expandedMinute", ev.get("minute", 0)),
            "period":          ev.get("period", {}).get("displayName", ""),
            "period_value":    ev.get("period", {}).get("value", 0),
            "event_type":      ev.get("type", {}).get("displayName", ""),
            "event_type_id":   ev.get("type", {}).get("value", 0),
            "outcome":         ev.get("outcomeType", {}).get("displayName", ""),
            "outcome_value":   ev.get("outcomeType", {}).get("value", 0),
            "is_touch":        int(ev.get("isTouch", False)),
            "team_id":         team_id,
            "is_home_team":    int(team_id == meta["home_team_id"]),
            "player_id":       player_id,
            "player_name":     p.get("name", ""),
            "player_position": p.get("position", ""),
            "x":               ev.get("x", 0.0),
            "y":               ev.get("y", 0.0),
            "end_x":           ev.get("endX"),
            "end_y":           ev.get("endY"),
            "is_throw_in":     int(qhas(qs, Q["THROW_IN"])),
            "is_long_ball":    int(qhas(qs, Q["LONG_BALL"])),
            "is_cross":        int(qhas(qs, Q["CROSS"])),
            "is_head":         int(qhas(qs, Q["HEAD_PASS"])),
            "is_freekick":     int(qhas(qs, Q["FREEKICK"])),
            "is_corner":       int(qhas(qs, Q["CORNER"])),
            "is_goal_kick":    int(qhas(qs, Q["GOAL_KICK"])),
            "is_through_ball": int(qhas(qs, Q["THROUGH_BALL"])),
            "is_key_pass":     int(qhas(qs, Q["KEY_PASS"])),
            "is_assist":       int(qhas(qs, Q["INTENT_ASSIST"]) or qhas(qs, Q["GOAL_ASSIST"])),
            "is_shot_assist":  int(qhas(qs, Q["SHOT_ASSIST"])),
            "is_offensive":    int(qhas(qs, Q["OFFENSIVE"])),
            "is_defensive":    int(qhas(qs, Q["DEFENSIVE"])),
            "is_last_man":     int(qhas(qs, Q["LAST_MAN"])),
            "pass_length":     qfloat(qs, Q["LENGTH"]),
            "pass_angle":      qfloat(qs, Q["ANGLE"]),
            "zone":            qval(qs, Q["ZONE"]) or "",
        })

    return pd.DataFrame(rows)


# ── 5. TEAM STATS ─────────────────────────────────────────────

def _build_team_stats(raw: dict, season: str) -> pd.DataFrame:
    mcd = raw["matchCentreData"]

    def total(stats: dict, key: str) -> int:
        d = stats.get(key, {})
        return int(sum(d.values())) if isinstance(d, dict) else 0

    def pct(num: int, den: int) -> float:
        return round(num / den * 100, 1) if den > 0 else 0.0

    rows = []

    # Calcular posesion desde toques de ambos equipos
    home_touches = total(mcd["home"].get("stats", {}), "touches")
    away_touches = total(mcd["away"].get("stats", {}), "touches")
    total_touches = home_touches + away_touches

    for tk in ["home", "away"]:
        team   = mcd[tk]
        stats  = team.get("stats", {})

        t_passes  = total(stats, "passesTotal")
        t_pass_ok = total(stats, "passesAccurate")
        t_aer     = total(stats, "aerialsTotal")
        t_aer_ok  = total(stats, "aerialsWon")
        t_tack    = total(stats, "tacklesTotal")
        t_tack_ok = total(stats, "tackleSuccessful")
        t_drib    = total(stats, "dribblesAttempted")
        t_drib_ok = total(stats, "dribblesWon")
        t_ti      = total(stats, "throwInsTotal")
        t_ti_ok   = total(stats, "throwInsAccurate")
        t_touches = total(stats, "touches")

        rows.append({
            "match_id":              raw["matchId"],
            "season":                season,
            "match_date":            mcd.get("startDate", ""),
            "team_id":               team["teamId"],
            "team_name":             team["name"],
            "opponent_id":           mcd["away"]["teamId"] if tk == "home" else mcd["home"]["teamId"],
            "opponent_name":         mcd["away"]["name"]   if tk == "home" else mcd["home"]["name"],
            "is_home":               int(tk == "home"),
            "result_score":          mcd.get("score", ""),
            "ft_score":              mcd.get("ftScore", ""),
            "ht_score":              mcd.get("htScore", ""),
            "venue":                 mcd.get("venueName", ""),
            "attendance":            mcd.get("attendance", 0),
            # Saques de banda (variable objetivo)
            "throw_ins_total":       t_ti,
            "throw_ins_accurate":    t_ti_ok,
            "throw_in_accuracy_pct": pct(t_ti_ok, t_ti),
            # Tiros
            "shots_total":           total(stats, "shotsTotal"),
            "shots_on_target":       total(stats, "shotsOnTarget"),
            "shots_off_target":      total(stats, "shotsOffTarget"),
            "shots_blocked":         total(stats, "shotsBlocked"),
            "shots_on_post":         total(stats, "shotsOnPost"),
            # Pases
            "passes_total":          t_passes,
            "passes_accurate":       t_pass_ok,
            "passes_key":            total(stats, "passesKey"),
            "pass_success_pct":      pct(t_pass_ok, t_passes),
            # Duelos aereos
            "aerials_total":         t_aer,
            "aerials_won":           t_aer_ok,
            "aerials_offensive":     total(stats, "offensiveAerials"),
            "aerials_defensive":     total(stats, "defensiveAerials"),
            "aerial_success_pct":    pct(t_aer_ok, t_aer),
            # Entradas
            "tackles_total":         t_tack,
            "tackles_successful":    t_tack_ok,
            "tackles_unsuccessful":  total(stats, "tackleUnsuccesful"),
            "tackle_success_pct":    pct(t_tack_ok, t_tack),
            "dribbled_past":         total(stats, "dribbledPast"),
            # Regates
            "dribbles_won":          t_drib_ok,
            "dribbles_attempted":    t_drib,
            "dribbles_lost":         total(stats, "dribblesLost"),
            "dribble_success_pct":   pct(t_drib_ok, t_drib),
            # Defensa
            "interceptions":         total(stats, "interceptions"),
            "clearances":            total(stats, "clearances"),
            "fouls_committed":       total(stats, "foulsCommited"),
            "dispossessed":          total(stats, "dispossessed"),
            # Corners
            "corners_total":         total(stats, "cornersTotal"),
            "corners_accurate":      total(stats, "cornersAccurate"),
            # Posesion y toques
            "touches_total":         t_touches,
            "possession_pct":        pct(t_touches, total_touches),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# GUARDADO LOCAL (CSV / Parquet)
# ─────────────────────────────────────────────────────────────

def save_df(df: pd.DataFrame, path: Path, fmt: str = "both"):
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt in ("csv", "both"):
        df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8")
    if fmt in ("parquet", "both"):
        try:
            df.to_parquet(path.with_suffix(".parquet"), index=False)
        except Exception as e:
            log.warning(f"No se pudo guardar parquet {path}: {e}")


def consolidate_outputs(out_dir: Path, season: str, fmt: str = "both"):
    log.info("Consolidando outputs...")
    tag = season.replace("/", "_")
    for name in ["throw_ins", "pass_map", "heatmap", "all_events", "team_stats"]:
        partials_dir = out_dir / "partials" / name
        files = list(partials_dir.glob("*.parquet")) or list(partials_dir.glob("*.csv"))
        if not files:
            continue
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f) if f.suffix == ".parquet"
                           else pd.read_csv(f))
            except Exception as e:
                log.warning(f"Error leyendo {f}: {e}")
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            out_path = out_dir / f"{tag}_{name}"
            save_df(combined, out_path, fmt)
            log.info(f"  ✓ {name}: {len(combined):,} filas → {out_path}")


# ─────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_scraper(
    season_key: str = "2025/2026",
    match_ids_override: list[int] = None,
    resume: bool = True,
    consolidate: bool = True,
    use_mongo: bool = True,
):
    """
    Pipeline completo: scraping + MongoDB + CSV/Parquet local.

    Args:
        season_key:           Temporada (ej: "2024/2025")
        match_ids_override:   Lista de IDs manuales (None = scrape fixtures)
        resume:               Salta partidos ya procesados
        consolidate:          Une parciales en un único archivo por tipo
        use_mongo:            Guarda en MongoDB Atlas en tiempo real
    """
    season_cfg = CONFIG["known_seasons"].get(season_key)
    if not season_cfg:
        raise ValueError(f"Temporada '{season_key}' no configurada. "
                         f"Disponibles: {list(CONFIG['known_seasons'].keys())}")

    tag     = season_key.replace("/", "_")
    out_dir = Path(CONFIG["output_dir"]) / tag
    raw_dir = out_dir / "raw"

    for name in ["throw_ins", "pass_map", "heatmap", "all_events", "team_stats"]:
        (out_dir / "partials" / name).mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(f"WhoScored LaLiga Scraper — {season_key}")
    log.info(f"Output: {out_dir.resolve()}")
    log.info("=" * 60)

    # Conectar a MongoDB
    mongo: MongoManager | None = None
    if use_mongo:
        try:
            mongo = MongoManager()
        except Exception as e:
            log.warning(f"MongoDB no disponible: {e}. Continuando solo con archivos.")

    # Resume: unir archivo local + Mongo (unión = no repetir partidos aunque cambies de modo)
    progress_file  = out_dir / "processed_ids.json"
    failed_file    = out_dir / "failed_ids.json"
    processed_ids: set[int] = set()
    failed_ids:    set[int] = set()
    if resume:
        if progress_file.exists():
            with open(progress_file, encoding="utf-8") as f:
                processed_ids |= set(json.load(f))
            log.info(f"  Archivo processed_ids.json: {len(processed_ids)} IDs")
        if failed_file.exists():
            with open(failed_file, encoding="utf-8") as f:
                failed_ids |= set(json.load(f))
            log.info(f"  Archivo failed_ids.json: {len(failed_ids)} IDs fallidos (se reintentarán)")
        if mongo:
            from_mongo = mongo.get_processed_ids(season=season_key)
            before = len(processed_ids)
            processed_ids |= from_mongo
            log.info(f"  MongoDB team_stats: +{len(processed_ids) - before} IDs (total {len(processed_ids)})")
        if processed_ids:
            log.info(f"Reanudando: {len(processed_ids)} partidos ya completos en {season_key}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=CONFIG["headless"],
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="es-ES",
            timezone_id="Europe/Madrid",
        )

        def _block(route, req):
            blocked_ext  = (".png", ".jpg", ".jpeg", ".gif", ".svg",
                            ".woff", ".woff2", ".ttf", ".mp4", ".webp")
            blocked_host = ("googlesyndication", "doubleclick", "facebook.com",
                            "google-analytics", "xtremepush", "ftd.agency",
                            "pubmatic", "openx.net", "3lift.com")
            if any(req.url.endswith(e) for e in blocked_ext):
                route.abort()
            elif any(d in req.url for d in blocked_host):
                route.abort()
            else:
                route.continue_()

        context.route("**/*", _block)
        page = context.new_page()

        # Paso 1: Match IDs
        if match_ids_override:
            match_ids = match_ids_override
            log.info(f"Usando {len(match_ids)} IDs manuales")
        else:
            match_ids = get_match_ids(page, season_cfg["season_id"],
                                      season_cfg["stage_id"])

        pending = [mid for mid in match_ids if mid not in processed_ids]
        log.info(f"Total: {len(match_ids)} | Pendientes: {len(pending)}")

        # Paso 2: Scrape
        fmt = CONFIG["output_format"]
        for match_id in tqdm(pending, desc=f"LaLiga {season_key}"):
            raw_path = raw_dir / f"{match_id}.json"
            raw = None

            if raw_path.exists():
                try:
                    with open(raw_path, encoding="utf-8") as f:
                        raw = json.load(f)
                except Exception:
                    raw = None

            if raw is None:
                raw = fetch_match_data(page, match_id)
                if raw:
                    with open(raw_path, "w", encoding="utf-8") as f:
                        json.dump(raw, f, ensure_ascii=False)
                    if mongo:
                        try:
                            mongo.save_raw(match_id, raw)
                        except OperationFailure as e:
                            log.warning(
                                "  MongoDB desactivado (cuota o error de escritura). "
                                "Continuando solo con archivos locales."
                            )
                            log.warning(f"  Detalle: {e}")
                            mongo = None
                        except Exception as e:
                            log.warning(f"  Raw no guardado en Mongo: {e}")
                else:
                    log.warning(f"  ✗ Sin datos: {match_id} — registrado en failed_ids.json")
                    failed_ids.add(match_id)
                    _save_failed(failed_file, failed_ids)
                    continue

            try:
                outputs  = process_match(raw, season_key)
                ts_df    = _build_team_stats(raw, season_key)

                # Guardar en MongoDB (en tiempo real)
                if mongo:
                    try:
                        mongo.save_throw_ins(outputs["throw_ins"])
                        mongo.save_pass_map(outputs["pass_map"])
                        mongo.save_heatmap(outputs["heatmap"])
                        mongo.save_all_events(outputs["all_events"])
                        mongo.save_team_stats(ts_df)
                    except OperationFailure as e:
                        log.warning(
                            "  MongoDB desactivado (cuota Atlas llena u otro error). "
                            "Siguientes partidos solo en disco."
                        )
                        log.warning(f"  Detalle: {e}")
                        mongo = None

                # Guardar parciales locales
                for name, df in outputs.items():
                    if not df.empty:
                        save_df(df, out_dir / "partials" / name / str(match_id), fmt)
                if not ts_df.empty:
                    save_df(ts_df, out_dir / "partials" / "team_stats" / str(match_id), fmt)

                log.info(
                    f"  ✓ {match_id} | "
                    f"throw_ins={len(outputs['throw_ins'])} | "
                    f"passes={len(outputs['pass_map'])} | "
                    f"touches={len(outputs['heatmap'])} | "
                    f"events={len(outputs['all_events'])}"
                )

            except OperationFailure as e:
                log.warning(f"  MongoDB: {e}")
                mongo = None
                processed_ids.add(match_id)
                _save_progress(progress_file, processed_ids)
            except Exception as e:
                log.error(f"  Error procesando {match_id}: {e}", exc_info=True)
                failed_ids.add(match_id)
                _save_failed(failed_file, failed_ids)
                random_delay()
                continue

            processed_ids.add(match_id)
            _save_progress(progress_file, processed_ids)
            random_delay()

        page.close()
        context.close()
        browser.close()

    if mongo:
        _print_mongo_summary(mongo)
        mongo.close()

    if consolidate:
        consolidate_outputs(out_dir, season_key, fmt)

    log.info("✅ Scraper finalizado")
    _print_file_summary(out_dir, season_key)


def _save_progress(path: Path, ids: set):
    with open(path, "w") as f:
        json.dump(list(ids), f)


def _save_failed(path: Path, ids: set):
    with open(path, "w") as f:
        json.dump(sorted(ids), f)


def _print_mongo_summary(mongo: MongoManager):
    log.info("\n─── MONGODB ─────────────────────────────────────────")
    for col, n in mongo.summary().items():
        log.info(f"  {col:<20} {n:>8,} documentos")
    log.info("─────────────────────────────────────────────────────")


def _print_file_summary(out_dir: Path, season: str):
    tag = season.replace("/", "_")
    log.info("\n─── ARCHIVOS LOCALES ────────────────────────────────")
    for name in ["throw_ins", "pass_map", "heatmap", "all_events", "team_stats"]:
        for ext in ["csv", "parquet"]:
            p = out_dir / f"{tag}_{name}.{ext}"
            if p.exists():
                size_mb = p.stat().st_size / 1_048_576
                log.info(f"  {p.name:<50s} {size_mb:6.1f} MB")
    log.info("─────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────
# VISUALIZACIONES (requiere: pip install mplsoccer matplotlib)
# ─────────────────────────────────────────────────────────────

def plot_heatmap(heatmap_df: pd.DataFrame, player_name: str = None,
                 team_name: str = None, save_path: str = None):
    """Genera un heatmap visual a partir del DataFrame de heatmap."""
    try:
        from mplsoccer import Pitch
        import matplotlib.pyplot as plt
    except ImportError:
        log.error("Instala: pip install mplsoccer matplotlib")
        return

    df = heatmap_df.copy()
    if player_name:
        df = df[df["player_name"] == player_name]
    if team_name:
        df = df[(df["home_team"] == team_name) | (df["away_team"] == team_name)]

    if df.empty:
        log.warning("Sin datos para el filtro")
        return

    # WhoScored [0,100]x[0,100] → StatsBomb [0,120]x[0,80]
    df["xp"] = df["x"] * 1.2
    df["yp"] = (100 - df["y"]) * 0.8

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2,
                  pitch_color="#22312b", line_color="white")
    fig, ax = pitch.draw(figsize=(10, 7))
    pitch.kdeplot(df["xp"], df["yp"], ax=ax, cmap="hot",
                  fill=True, levels=100, alpha=0.7)

    ax.set_title(f"Mapa de Actividad — {player_name or team_name or 'General'}",
                 color="white", fontsize=14)
    fig.set_facecolor("#22312b")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Heatmap guardado: {save_path}")
    else:
        plt.show()


def plot_pass_map(pass_map_df: pd.DataFrame, player_name: str = None,
                  team_name: str = None, only_throw_ins: bool = False,
                  save_path: str = None):
    """Genera un mapa de pases a partir del DataFrame de pass_map."""
    try:
        from mplsoccer import Pitch
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        log.error("Instala: pip install mplsoccer matplotlib")
        return

    df = pass_map_df.copy()
    if player_name:
        df = df[df["player_name"] == player_name]
    if team_name:
        mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
        df   = df[mask]
        if not df.empty:
            is_home = df["home_team"].iloc[0] == team_name
            df = df[df["is_home_team"] == int(is_home)]
    if only_throw_ins:
        df = df[df["is_throw_in"] == 1]

    if df.empty:
        log.warning("Sin datos para el filtro")
        return

    df["xs"]    = df["x"]     * 1.2
    df["ys"]    = (100 - df["y"])     * 0.8
    df["end_xs"]= df["end_x"] * 1.2
    df["end_ys"]= (100 - df["end_y"]) * 0.8

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2,
                  pitch_color="#1a1a2e", line_color="white")
    fig, ax = pitch.draw(figsize=(12, 8))

    # Vectorizado: una llamada pitch.arrows por grupo (éxito × throw-in)
    for is_ok, is_ti, color, alpha, lw in [
        (True,  True,  "#00ff88", 0.7, 2.5),
        (True,  False, "#00ff88", 0.7, 1.0),
        (False, True,  "#ff4444", 0.4, 2.5),
        (False, False, "#ff4444", 0.4, 1.0),
    ]:
        grp = df[(df["is_successful"] == int(is_ok)) & (df["is_throw_in"] == int(is_ti))]
        if grp.empty:
            continue
        pitch.arrows(grp["xs"].values, grp["ys"].values,
                     grp["end_xs"].values, grp["end_ys"].values,
                     ax=ax, color=color, alpha=alpha,
                     width=lw, headwidth=3, headlength=4)

    title = " | ".join(filter(None, [
        player_name, team_name,
        "Solo saques de banda" if only_throw_ins else None,
    ])) or "Mapa de Pases"
    ax.set_title(f"Pass Map — {title}", color="white", fontsize=14)
    fig.set_facecolor("#1a1a2e")
    ax.legend(
        handles=[
            mpatches.Patch(color="#00ff88", label="Completado"),
            mpatches.Patch(color="#ff4444", label="Fallido"),
        ],
        loc="lower right", facecolor="#1a1a2e", labelcolor="white",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Pass map guardado: {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    setup_logging("scraper.log")

    parser = argparse.ArgumentParser(
        description="WhoScored LaLiga Scraper — saques de banda + heatmaps + pass maps + MongoDB"
    )
    parser.add_argument("--season",     default="2025/2026",
                        help="Temporada (ej: 2024/2025)")
    parser.add_argument("--match-ids",  nargs="+", type=int, default=None,
                        help="Match IDs específicos")
    parser.add_argument("--no-resume",  action="store_true",
                        help="Reiniciar desde cero")
    parser.add_argument("--no-headless",action="store_true",
                        help="Mostrar navegador")
    parser.add_argument("--no-mongo",   action="store_true",
                        help="Desactivar MongoDB")
    parser.add_argument("--format",     choices=["csv", "parquet", "both"],
                        default="both", help="Formato de salida local")
    args = parser.parse_args()

    CONFIG["headless"]      = not args.no_headless
    CONFIG["output_format"] = args.format

    run_scraper(
        season_key         = args.season,
        match_ids_override = args.match_ids,
        resume             = not args.no_resume,
        consolidate        = True,
        use_mongo          = not args.no_mongo,
    )
