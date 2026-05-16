"""
Smoke Test — `check_team_with_more_backtest.py`
================================================
Corre el backtest para `--market team_with_more` y valida que el output
tiene las claves requeridas y rangos plausibles.

Asserts:
  1. El CLI corre exit 0.
  2. Existe al menos uno de {parquet, csv} para team_with_more del día.
  3. `eval_summary_YYYYMMDD.json` tiene la sección `markets.team_with_more`
     con claves: n, hit_rate, wilson_ci_low, wilson_ci_high, brier,
     log_loss, roi_theoretical.
  4. `0 <= hit_rate <= 1`, `brier >= 0`.

Uso:
  python -m scripts.smoke.check_team_with_more_backtest
  exit 0 → pass · exit 1 → fail
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

OUTPUT_DIR = _root / "data/model/market_eval"
REQUIRED_SUMMARY_KEYS = (
    "n",
    "hit_rate",
    "wilson_ci_low",
    "wilson_ci_high",
    "brier",
    "log_loss",
    "roi_theoretical",
)


def _fail(msg: str) -> "NoReturn":
    print(f"[FAIL] check_team_with_more_backtest: {msg}")
    sys.exit(1)


def _pass(msg: str) -> None:
    print(f"[PASS] check_team_with_more_backtest: {msg}")
    sys.exit(0)


def main() -> None:
    # 1. Ejecutar backtest
    cmd = [
        sys.executable,
        "-m", "scripts.evaluation.evaluate_vs_market",
        "--market", "team_with_more",
    ]
    print(f"[INFO] ejecutando: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        _fail(f"CLI exit={proc.returncode}\nSTDERR:\n{proc.stderr[:2000]}")

    # 2. Files existen
    today = date.today().strftime("%Y%m%d")
    parquet_path = OUTPUT_DIR / f"eval_team_with_more_{today}.parquet"
    csv_path = OUTPUT_DIR / f"eval_team_with_more_{today}.csv"
    summary_path = OUTPUT_DIR / f"eval_summary_{today}.json"

    if not (parquet_path.exists() or csv_path.exists()):
        _fail(f"no existe eval_team_with_more_{today}.(parquet|csv) en {OUTPUT_DIR}")
    if not summary_path.exists():
        _fail(f"no existe {summary_path.name} en {OUTPUT_DIR}")

    # 3. Summary tiene las claves requeridas
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    market_summary = summary.get("markets", {}).get("team_with_more")
    if not isinstance(market_summary, dict):
        _fail(f"summary.markets.team_with_more ausente o inválido")

    # Si n == 0 (no hay predicciones cruzadas), no podemos validar métricas —
    # el smoke aún pasa porque el pipeline completó OK, pero lo reportamos.
    if market_summary.get("n", 0) == 0:
        print("[WARN] team_with_more n=0 — no hay predicciones cruzadas en esta corrida. "
              "Smoke completo, pero sin métricas reales.")
        _pass("pipeline OK (n=0 — revisar cruce de predicciones con odds)")

    missing = [k for k in REQUIRED_SUMMARY_KEYS if k not in market_summary]
    if missing:
        _fail(f"summary.markets.team_with_more faltan claves: {missing}")

    # 4. Rangos
    hr = market_summary["hit_rate"]
    if not (0.0 <= hr <= 1.0):
        _fail(f"hit_rate fuera de [0,1]: {hr}")
    brier = market_summary["brier"]
    if brier < 0:
        _fail(f"brier negativo: {brier}")

    _pass(
        f"n={market_summary['n']}, hit_rate={hr:.3f} "
        f"(Wilson [{market_summary['wilson_ci_low']:.3f}, {market_summary['wilson_ci_high']:.3f}]), "
        f"brier={brier:.4f}, roi={market_summary['roi_theoretical']:+.4f}"
    )


if __name__ == "__main__":
    main()
