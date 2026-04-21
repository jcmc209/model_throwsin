"""
Referee Extractor
=================
Parsea los raw JSONs de WhoScored ya descargados y extrae el árbitro
de cada partido junto con el total de saques de banda.

Output:
  data/reference/referee_stats.parquet
    match_id               int64
    referee_id             int64
    referee_name           str
    match_date             datetime64[ns]
    throw_ins_total_match  int32     ← suma home + away

Uso:
  python scripts/referee_extractor.py
  python scripts/referee_extractor.py --output data/reference/referee_stats.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("referee_extractor")

CONFIG = {
    "raw_glob": "data/whoscored_laliga/*/raw/*.json",
    "team_stats_glob": "data/whoscored_laliga/**/*_team_stats.parquet",
    "output_path": "data/reference/referee_stats.parquet",
}


def extract_referee_from_jsons() -> pd.DataFrame:
    """Extrae match_id, referee_id, referee_name, match_date de cada raw JSON."""
    raw_files = sorted(glob.glob(CONFIG["raw_glob"]))
    if not raw_files:
        raise FileNotFoundError(f"No se encontraron raw JSONs en {CONFIG['raw_glob']}")
    log.info("Procesando %d raw JSONs ...", len(raw_files))

    records = []
    errors = 0
    for path in raw_files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            mcd = data["matchCentreData"]
            ref = mcd.get("referee") or {}
            records.append({
                "match_id": int(data["matchId"]),
                "referee_id": ref.get("officialId"),
                "referee_name": ref.get("name", ""),
                "match_date": pd.to_datetime(mcd.get("startDate") or mcd.get("timeStamp")),
            })
        except Exception as exc:
            log.warning("Error en %s: %s", path, exc)
            errors += 1

    if errors:
        log.warning("%d archivos con errores (ignorados)", errors)

    df = pd.DataFrame(records)
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.normalize()
    df["referee_id"] = pd.to_numeric(df["referee_id"], errors="coerce")
    return df


def compute_throw_ins_per_match() -> pd.DataFrame:
    """Suma throw_ins_total home + away por match_id desde team_stats.parquet."""
    files = sorted(glob.glob(CONFIG["team_stats_glob"], recursive=True))
    if not files:
        raise FileNotFoundError(f"No se encontraron team_stats en {CONFIG['team_stats_glob']}")

    ts = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    per_match = (
        ts.groupby("match_id")["throw_ins_total"]
        .sum()
        .reset_index()
        .rename(columns={"throw_ins_total": "throw_ins_total_match"})
    )
    per_match["throw_ins_total_match"] = per_match["throw_ins_total_match"].astype("int32")
    return per_match


def build_referee_stats() -> pd.DataFrame:
    referee_df = extract_referee_from_jsons()
    throw_ins_df = compute_throw_ins_per_match()

    df = referee_df.merge(throw_ins_df, on="match_id", how="left")

    nulls_ref = df["referee_id"].isna().sum()
    nulls_ti = df["throw_ins_total_match"].isna().sum()
    if nulls_ref:
        log.warning("%d partidos sin referee_id", nulls_ref)
    if nulls_ti:
        log.warning("%d partidos sin throw_ins_total_match", nulls_ti)

    df["referee_id"] = df["referee_id"].astype("Int64")

    log.info(
        "referee_stats: %d filas | %d árbitros únicos | %d nulls referee_id",
        len(df), df["referee_id"].nunique(), nulls_ref,
    )
    return df[["match_id", "referee_id", "referee_name", "match_date", "throw_ins_total_match"]]


def main(output_path: str | None = None) -> None:
    out = Path(output_path or CONFIG["output_path"])
    df = build_referee_stats()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Guardado: %s (%d filas)", out, len(df))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Extract referee data from raw JSONs")
    parser.add_argument("--output", default=CONFIG["output_path"])
    args = parser.parse_args()
    main(output_path=args.output)
