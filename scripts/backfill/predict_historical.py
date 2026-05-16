"""
Backfill de predicciones históricas — Throw-In Predictor
========================================================
Genera `data/model/predictions/predictions_YYYYMMDD.parquet` para fechas
pasadas de la temporada activa, usando el modelo actual (`model_v1.joblib`)
para todas las fechas. Este NO es un walk-forward real: es "qué predeciría
el modelo actual sobre fixtures pasados". Aceptado como limitación honesta
para poder correr backtest histórico vs Codere.

Flujo:
  1. Lee `data/model/dataset.parquet` → fechas 2025/26 con partidos jugados.
  2. Opcionalmente intersecta con el universo de matchups presentes en
     `data/reference/odds_codere.parquet` para backfillear solo fechas útiles.
  3. Para cada fecha sin parquet correspondiente en `data/model/predictions/`,
     fuerza el `status='scheduled'` sobre el calendario (temporalmente) y llama
     a `model.predict.main(date_filter=date)` escribiendo en un tmp dir.
  4. Renombra el output `predictions_{TODAY}.parquet` → `predictions_{date}.parquet`.
  5. Loggea progreso por fecha (ok / fail <motivo>).

Uso:
  python -m scripts.backfill.predict_historical
  python -m scripts.backfill.predict_historical --dates 2025-09-28 2025-10-25
  python -m scripts.backfill.predict_historical --only-codere-overlap

Idempotente: si `predictions_{date}.parquet` ya existe, se salta la fecha.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import unicodedata
from datetime import date as _date
from pathlib import Path

import pandas as pd

# ── path setup ───────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model import predict as predict_mod
from model.market_utils import normalize_team
from scripts.evaluation._normalize import _normalize_team_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("backfill_predict")

PRED_DIR = _root / "data/model/predictions"
DATASET = _root / "data/model/dataset.parquet"
CALENDAR = _root / "data/reference/liga_calendar_rows.csv"
ODDS_CODERE = _root / "data/reference/odds_codere.parquet"


def _list_codere_pairs() -> set[tuple[str, str]]:
    """Devuelve el set de (home_ds, away_ds) normalizados presentes en Codere."""
    co = pd.read_parquet(ODDS_CODERE)
    pairs = set()
    for row in co.itertuples():
        h = _normalize_team_name(normalize_team(row.home_team, "codere_to_ds"))
        a = _normalize_team_name(normalize_team(row.away_team, "codere_to_ds"))
        pairs.add((h, a))
    return pairs


def _pick_dates(only_codere_overlap: bool, explicit: list[str] | None) -> list[str]:
    """Selecciona las fechas a backfillear (formato YYYY-MM-DD)."""
    if explicit:
        return sorted(set(explicit))

    ds = pd.read_parquet(DATASET)
    ds26 = ds[ds["season"] == "2025/2026"].copy()
    ds26_home = ds26[ds26["is_home"] == 1].copy()
    ds26_home["match_date"] = pd.to_datetime(ds26_home["match_date"]).dt.date
    today = _date.today()
    ds26_home = ds26_home[ds26_home["match_date"] < today]

    if only_codere_overlap:
        codere_pairs = _list_codere_pairs()
        ds26_home["hn"] = ds26_home["team_name"].map(_normalize_team_name)
        ds26_home["an"] = ds26_home["opponent_name"].map(_normalize_team_name)
        mask = ds26_home.apply(lambda r: (r["hn"], r["an"]) in codere_pairs, axis=1)
        ds26_home = ds26_home[mask]

    dates = sorted({d.strftime("%Y-%m-%d") for d in ds26_home["match_date"].unique()})
    return dates


def _force_calendar_scheduled(target_date: str) -> pd.DataFrame:
    """Lee el calendario y retorna una copia donde los partidos de `target_date`
    tienen `status='scheduled'` (para que `predict.load_scheduled_matches` los
    incluya). No se persiste: solo devuelve el DataFrame modificado."""
    cal = pd.read_csv(CALENDAR)
    cal_dates = pd.to_datetime(cal["match_date"]).dt.strftime("%Y-%m-%d")
    mask = cal_dates == target_date
    if not mask.any():
        return cal  # nada que tocar, predict devolverá vacío
    cal.loc[mask, "status"] = "scheduled"
    return cal


def _backfill_one(target_date: str, pred_dir: Path) -> tuple[str, str]:
    """Backfillea una fecha. Retorna (status, detail).

    status ∈ {'ok', 'skip', 'fail'}.
    """
    compact = target_date.replace("-", "")
    out_path = pred_dir / f"predictions_{compact}.parquet"
    if out_path.exists():
        return ("skip", f"{out_path.name} ya existe")

    # Directorio temporal por fecha para aislar el output
    tmp_dir = pred_dir.parent / f"_tmp_backfill_{compact}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Parchar el calendario en memoria: cargar, marcar 'scheduled' sobre la
    # fecha objetivo, y reemplazar la función `load_scheduled_matches` en runtime
    # para que use nuestra versión parcheada. Esto evita escribir sobre el CSV.
    original_loader = predict_mod.load_scheduled_matches
    forced_cal = _force_calendar_scheduled(target_date)

    def _patched_loader(date_filter, matchday_next, all_scheduled):
        cal = forced_cal.copy()
        cal["match_date"] = pd.to_datetime(cal["match_date"])
        cal = cal[
            (cal["status"] == "scheduled")
            & (cal["competition"].str.contains("La Liga", case=False, na=False))
        ].copy()
        if date_filter:
            target = pd.to_datetime(date_filter).normalize()
            cal = cal[cal["match_date"] == target]
        if "referee_name" not in cal.columns:
            cal["referee_name"] = None
        else:
            cal["referee_name"] = cal["referee_name"].replace("", None).where(
                cal["referee_name"].notna() & (cal["referee_name"].str.strip() != ""), None
            )
        return cal.reset_index(drop=True)

    try:
        predict_mod.load_scheduled_matches = _patched_loader  # type: ignore[assignment]
        predict_mod.main(
            date_filter=target_date,
            matchday_next=False,
            all_scheduled=False,
            output_path=str(tmp_dir),
        )
    except SystemExit as e:
        predict_mod.load_scheduled_matches = original_loader  # type: ignore[assignment]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return ("fail", f"SystemExit({e.code})")
    except Exception as e:
        predict_mod.load_scheduled_matches = original_loader  # type: ignore[assignment]
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return ("fail", f"{type(e).__name__}: {e}")
    finally:
        predict_mod.load_scheduled_matches = original_loader  # type: ignore[assignment]

    # El archivo generado se llama predictions_{TODAY}.parquet (predict.py usa
    # datetime.utcnow()). Lo ubicamos y renombramos.
    produced = sorted(tmp_dir.glob("predictions_*.parquet"))
    if not produced:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return ("fail", "predict no generó parquet (posible sin-fixture)")

    src = produced[0]
    shutil.move(str(src), str(out_path))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return ("ok", f"{out_path.name} ({out_path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill de predicciones históricas")
    parser.add_argument(
        "--dates", nargs="+", default=None,
        help="Lista explícita de fechas YYYY-MM-DD. Si se omite, se derivan del dataset.",
    )
    parser.add_argument(
        "--only-codere-overlap", action="store_true",
        help="Solo backfillear fechas cuyos matchups coincidan con odds_codere.parquet.",
    )
    args = parser.parse_args()

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    dates = _pick_dates(args.only_codere_overlap, args.dates)
    log.info("Fechas a procesar: %d", len(dates))

    summary = {"ok": [], "skip": [], "fail": []}
    for i, d in enumerate(dates, 1):
        log.info("[%d/%d] backfilling %s ...", i, len(dates), d)
        status, detail = _backfill_one(d, PRED_DIR)
        summary[status].append((d, detail))
        log.info("[%d/%d] date=%s %s %s", i, len(dates), d, status, detail)

    log.info("=" * 60)
    log.info("RESUMEN BACKFILL")
    log.info("  ok  : %d", len(summary["ok"]))
    log.info("  skip: %d", len(summary["skip"]))
    log.info("  fail: %d", len(summary["fail"]))
    if summary["fail"]:
        log.warning("Fechas con fallo:")
        for d, detail in summary["fail"]:
            log.warning("  %s → %s", d, detail)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
