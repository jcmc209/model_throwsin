"""
Smoke Test — `check_bias_applied.py`
=====================================
Valida que las predicciones más recientes incluyen columnas v2 con
valores plausibles (lo que indica que `apply_team_bias` corrió en inferencia).

Asserts mínimos (sin λ raw en el parquet no podemos recomputar el bias
exacto, así que usamos rangos de plausibilidad + que `load_team_bias()`
devuelva una tabla no-vacía):
  1. Último parquet en `data/model/predictions/` existe y tiene las 3 v2 cols.
  2. Ningún NaN en v2 cols.
  3. Per-team λ ∈ [5, 25] (throw-ins por equipo por partido — rango histórico LaLiga).
  4. λ total ∈ [15, 45].
  5. `load_team_bias()` devuelve dict no vacío (la tabla está disponible).

Uso:
  python -m scripts.smoke.check_bias_applied
  exit 0 → pass · exit 1 → fail
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.market_utils import load_team_bias

PRED_DIR = _root / "data/model/predictions"
V2_COLS = ("pred_home_v2", "pred_away_v2", "pred_total_v2")
PER_TEAM_RANGE = (5.0, 25.0)
TOTAL_RANGE = (15.0, 45.0)


def _fail(msg: str) -> "NoReturn":
    print(f"[FAIL] check_bias_applied: {msg}")
    sys.exit(1)


def _pass(msg: str) -> None:
    print(f"[PASS] check_bias_applied: {msg}")
    sys.exit(0)


def main() -> None:
    files = sorted(PRED_DIR.glob("predictions_*.parquet"))
    if not files:
        _fail(f"no hay parquets en {PRED_DIR}")

    pred_file = files[-1]
    df = pd.read_parquet(pred_file)
    n = len(df)
    if n == 0:
        _fail(f"{pred_file.name} está vacío")

    # 1. columnas presentes
    missing = [c for c in V2_COLS if c not in df.columns]
    if missing:
        _fail(f"columnas faltantes en {pred_file.name}: {missing}")

    # 2. sin NaN en v2
    for c in V2_COLS:
        if df[c].isna().any():
            _fail(f"{c} contiene NaN en {pred_file.name}")

    # 3. rango por equipo (home/away)
    for c in ("pred_home_v2", "pred_away_v2"):
        lo, hi = PER_TEAM_RANGE
        out_of_range = df[(df[c] < lo) | (df[c] > hi)]
        if len(out_of_range) > 0:
            _fail(
                f"{c}: {len(out_of_range)} filas fuera de rango [{lo}, {hi}] — "
                f"min={df[c].min():.2f}, max={df[c].max():.2f}"
            )

    # 4. rango total
    lo, hi = TOTAL_RANGE
    out_of_range = df[(df["pred_total_v2"] < lo) | (df["pred_total_v2"] > hi)]
    if len(out_of_range) > 0:
        _fail(
            f"pred_total_v2: {len(out_of_range)} filas fuera de rango [{lo}, {hi}] — "
            f"min={df['pred_total_v2'].min():.2f}, max={df['pred_total_v2'].max():.2f}"
        )

    # 5. tabla de bias cargable y no vacía
    bias = load_team_bias()
    if not isinstance(bias, dict) or len(bias) == 0:
        _fail("load_team_bias() devolvió dict vacío — calibración no disponible")

    # 6. identidad: total ≈ home + away
    diff = np.abs(df["pred_total_v2"].values
                  - df["pred_home_v2"].values
                  - df["pred_away_v2"].values)
    max_diff = float(diff.max())
    if max_diff > 1e-6:
        _fail(f"|total − (home+away)| max = {max_diff:.3e} > 1e-6")

    per_team_min = float(df[["pred_home_v2", "pred_away_v2"]].min().min())
    per_team_max = float(df[["pred_home_v2", "pred_away_v2"]].max().max())
    total_min = float(df["pred_total_v2"].min())
    total_max = float(df["pred_total_v2"].max())
    _pass(
        f"{pred_file.name} | n={n} | v2 ok | "
        f"per-team [{per_team_min:.1f}, {per_team_max:.1f}] | "
        f"total [{total_min:.1f}, {total_max:.1f}] | "
        f"bias_table n_teams={len(bias)}"
    )


if __name__ == "__main__":
    main()
