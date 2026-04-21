"""
Smoke — columnas v2 en predictions parquet
===========================================
Asserta que el último `predictions_*.parquet` tiene las 3 columnas calibradas
(`pred_home_v2`, `pred_away_v2`, `pred_total_v2`), son float64, no-null, y
`pred_total_v2 ≈ pred_home_v2 + pred_away_v2` (tolerancia 1e-6).

Uso:
  python -m scripts.smoke.check_v2_columns
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("pred_home_v2", "pred_away_v2", "pred_total_v2")
TOL = 1e-6
PRED_DIR = Path("data/model/predictions")


def _latest_predictions_parquet() -> Path | None:
    if not PRED_DIR.exists():
        return None
    candidates = sorted(PRED_DIR.glob("predictions_*.parquet"))
    if not candidates:
        return None
    return candidates[-1]


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    path = _latest_predictions_parquet()
    if path is None:
        _fail(f"no predictions_*.parquet en {PRED_DIR}/ — corré `python -m model.predict` primero")

    df = pd.read_parquet(path)
    if df.empty:
        _fail(f"parquet vacío: {path}")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        _fail(f"columnas ausentes en {path.name}: {missing} (presentes: {list(df.columns)})")

    for col in REQUIRED:
        if df[col].dtype != np.float64:
            _fail(f"{col} dtype={df[col].dtype}, esperado float64")
        if df[col].isna().any():
            n_null = int(df[col].isna().sum())
            _fail(f"{col} tiene {n_null} nulos en {path.name}")

    diff = (df["pred_total_v2"] - (df["pred_home_v2"] + df["pred_away_v2"])).abs()
    max_diff = float(diff.max())
    if max_diff > TOL:
        _fail(
            f"pred_total_v2 != pred_home_v2 + pred_away_v2 (max|diff|={max_diff:.3e} > {TOL}) en {path.name}"
        )

    print(
        f"PASS: {path.name} — {len(df)} filas, v2 columns ok, max|total-home-away|={max_diff:.3e}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
