"""
PostgreSQL helper — odds storage and scheduler state.

Active when DATABASE_URL env var is set (Railway deployment).
Local dev sigue usando parquet + JSON sin cambios.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

import pandas as pd

log = logging.getLogger(__name__)

DATABASE_URL: str | None = os.environ.get("DATABASE_URL")

_DDL = """
CREATE TABLE IF NOT EXISTS odds_22bet (
    id              SERIAL           PRIMARY KEY,
    match_ci        BIGINT           NOT NULL,
    home_team       TEXT,
    away_team       TEXT,
    match_date      TIMESTAMPTZ,
    scraped_at      TIMESTAMPTZ      NOT NULL,
    hours_before    DOUBLE PRECISION,
    market_type     TEXT,
    line            DOUBLE PRECISION NOT NULL DEFAULT -1,
    side            TEXT,
    odds            DOUBLE PRECISION,
    bookmaker       TEXT             DEFAULT '22bet',
    raw_market_name TEXT,
    raw_selection   TEXT,
    UNIQUE (match_ci, scraped_at, market_type, line, side)
);

CREATE TABLE IF NOT EXISTS scheduler_state (
    state_key    TEXT        PRIMARY KEY,
    triggered_at TIMESTAMPTZ NOT NULL
);
"""


@contextmanager
def _connection() -> Generator:
    import psycopg2
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def is_available() -> bool:
    return bool(DATABASE_URL)


def init_tables() -> None:
    """Crea las tablas si no existen. Llamar al arranque del scheduler."""
    if not is_available():
        return
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
    log.info("PostgreSQL: tablas listas.")


def upsert_odds(df: pd.DataFrame) -> None:
    """Inserta cuotas en PostgreSQL. Duplicados (mismo match/scraped_at/market/line/side) se ignoran."""
    if not is_available() or df.empty:
        return

    import psycopg2.extras

    cols = [
        "match_ci", "home_team", "away_team", "match_date", "scraped_at",
        "hours_before", "market_type", "line", "side", "odds",
        "bookmaker", "raw_market_name", "raw_selection",
    ]
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = None

    # NULL line (team_with_more) → sentinel -1.0 para satisfacer la UNIQUE constraint
    df["line"] = df["line"].fillna(-1.0)

    records = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]

    sql = f"""
        INSERT INTO odds_22bet ({", ".join(cols)})
        VALUES ({", ".join(["%s"] * len(cols))})
        ON CONFLICT (match_ci, scraped_at, market_type, line, side)
        DO NOTHING
    """
    with _connection() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, records)
    log.info("PostgreSQL: %d filas insertadas en odds_22bet.", len(records))


def load_state() -> dict:
    """Devuelve el estado del scheduler desde PostgreSQL."""
    if not is_available():
        return {}
    with _connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT state_key, triggered_at FROM scheduler_state")
            return {row[0]: row[1].isoformat() for row in cur.fetchall()}


def save_state(state: dict) -> None:
    """Upserta todas las entradas del estado en PostgreSQL."""
    if not is_available() or not state:
        return
    with _connection() as conn:
        with conn.cursor() as cur:
            for key, val in state.items():
                cur.execute(
                    """
                    INSERT INTO scheduler_state (state_key, triggered_at)
                    VALUES (%s, %s)
                    ON CONFLICT (state_key)
                    DO UPDATE SET triggered_at = EXCLUDED.triggered_at
                    """,
                    (key, val),
                )
    log.info("PostgreSQL: estado del scheduler guardado (%d entradas).", len(state))
