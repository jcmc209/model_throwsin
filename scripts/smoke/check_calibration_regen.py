"""
Smoke Test — `check_calibration_regen.py`
=========================================
Compara la tabla de calibración recién regenerada (dry-run) contra la actualmente
en producción (`data/model/team_bias_calibration_v2.json`). Sirve como GATE antes
de sobrescribir la versión de producción.

Flujo:
  1. Carga joblib actual.
  2. Corre `_regenerate_team_bias_calibration` escribiendo a un path temporal.
  3. Carga el JSON temporal y el de producción.
  4. Por (team_id, is_home) calcula delta = shrunk_regen - shrunk_prod.
  5. Reporta mean |delta|, max |delta|, teams que cambian de signo.
  6. Exit 0 si mean|delta| < THRESHOLD_MEAN y max|delta| < THRESHOLD_MAX; si no, exit 1.

Thresholds (generosos por diseño — el modelo cambia de vez en cuando):
  THRESHOLD_MEAN = 0.5 (throws)  — promedio del shift permitido
  THRESHOLD_MAX  = 2.0 (throws)  — individual team shift máximo

El umbral más estricto tripea cuando el modelo cambia significativamente;
esa es una señal diagnóstica válida — no seguir sobrescribiendo, inspeccionar.

Uso:
  python -m scripts.smoke.check_calibration_regen
  exit 0 → pass · exit 1 → fail
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.train import CONFIG, _regenerate_team_bias_calibration

PROD_JSON = _root / CONFIG["team_bias_path"]
THRESHOLD_MEAN = 0.5
THRESHOLD_MAX = 2.0


def _fail(msg: str) -> "NoReturn":
    print(f"[FAIL] check_calibration_regen: {msg}")
    sys.exit(1)


def _pass(msg: str) -> None:
    print(f"[PASS] check_calibration_regen: {msg}")
    sys.exit(0)


def _flatten(corrections: dict) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for tid, byh in corrections.items():
        for hid, stats in byh.items():
            out[(str(tid), str(hid))] = float(stats["shrunk_bias"])
    return out


def main() -> None:
    # 1. Cargar producción.
    if not PROD_JSON.exists():
        _fail(f"no existe JSON de producción {PROD_JSON}")
    with open(PROD_JSON, "r", encoding="utf-8") as f:
        prod_payload = json.load(f)
    prod_flat = _flatten(prod_payload.get("corrections", {}))
    if not prod_flat:
        _fail("JSON de producción sin corrections")

    # 2. Cargar joblib y val.
    model_p = _root / CONFIG["model_path"]
    if not model_p.exists():
        _fail(f"no existe joblib {model_p}")
    artifact = joblib.load(model_p)
    val_seasons = list(artifact.get("val_seasons", []))
    if not val_seasons:
        _fail("artifact['val_seasons'] vacío")

    df = pd.read_parquet(_root / CONFIG["dataset_path"])
    val = df[df["season"].isin(val_seasons)].copy()
    if val.empty:
        _fail(f"val vacío para val_seasons={val_seasons}")

    # 3. Regenerar a tmp paths (aislados — ni tocar prod ni backup).
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / "regen.json"
        tmp_bak = Path(tmp_dir) / "never_used_bak.json"
        _regenerate_team_bias_calibration(
            model=artifact["model"],
            val_df=val,
            feature_cols=list(artifact["features"]),
            model_trained_at=artifact.get("trained_at"),
            model_train_seasons=list(artifact.get("train_seasons", [])),
            out_path=tmp_out,
            backup_path=tmp_bak,
        )
        with open(tmp_out, "r", encoding="utf-8") as f:
            regen_payload = json.load(f)
    regen_flat = _flatten(regen_payload.get("corrections", {}))

    # 4. Schema / keys.
    if set(regen_flat.keys()) != set(prod_flat.keys()):
        only_prod = sorted(set(prod_flat) - set(regen_flat))
        only_regen = sorted(set(regen_flat) - set(prod_flat))
        _fail(
            f"claves no coinciden — "
            f"only_prod[:3]={only_prod[:3]} ({len(only_prod)}), "
            f"only_regen[:3]={only_regen[:3]} ({len(only_regen)})"
        )

    # 5. Delta table.
    deltas = []
    sign_flips = []
    for key, prod_val in prod_flat.items():
        regen_val = regen_flat[key]
        delta = regen_val - prod_val
        deltas.append(abs(delta))
        if prod_val * regen_val < 0:
            sign_flips.append((key, prod_val, regen_val))

    if not deltas:
        _fail("sin pares para comparar (deltas vacío)")

    mean_abs = sum(deltas) / len(deltas)
    max_abs = max(deltas)

    # 6. Checks + report.
    issues = []
    if mean_abs >= THRESHOLD_MEAN:
        issues.append(f"mean|delta|={mean_abs:.4f} >= {THRESHOLD_MEAN}")
    if max_abs >= THRESHOLD_MAX:
        issues.append(f"max|delta|={max_abs:.4f} >= {THRESHOLD_MAX}")

    msg = (
        f"n_pairs={len(deltas)} mean|delta|={mean_abs:.4f} max|delta|={max_abs:.4f} "
        f"sign_flips={len(sign_flips)}"
    )
    if issues:
        _fail(msg + " | " + "; ".join(issues))
    _pass(msg)


if __name__ == "__main__":
    main()
