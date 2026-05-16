"""
Backfill — Regenerar team_bias_calibration_v2.json sin reentrenar el modelo
===========================================================================
Carga el joblib actual (`data/model/model_v1.joblib`), aplica predict sobre el
split de validación (`val_seasons` del artifact) y regenera
`data/model/team_bias_calibration_v2.json` reutilizando
`model.train._regenerate_team_bias_calibration`.

Por qué Option B y no reentrenar:
  - El followup #28 no cambia el modelo — solo cierra el orphan-write del JSON.
  - Evita perder minutos en reentrenar cuando lo único que queremos regenerar
    es el artifact de residuos (que depende SOLO de la pareja model + val).
  - El single code path queda preservado: el regen invoca el mismo helper
    que `model/train.py:main()` invoca al final.

Uso:
  python -m scripts.backfill.regen_calibration_from_existing_model
  python -m scripts.backfill.regen_calibration_from_existing_model --dry-run
  python -m scripts.backfill.regen_calibration_from_existing_model --out /tmp/foo.json

Exit 0 en éxito, != 0 en error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from model.train import CONFIG, _regenerate_team_bias_calibration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("regen_calibration")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=CONFIG["model_path"],
        help="Ruta al joblib del modelo (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-path",
        default=CONFIG["dataset_path"],
        help="Ruta al dataset.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override del output JSON (default: CONFIG['team_bias_path']). "
             "Útil para dry-run sin tocar el archivo de producción.",
    )
    parser.add_argument(
        "--backup",
        default=None,
        help="Override del backup path (default: CONFIG['team_bias_pre_regen_bak_path']).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Escribe a un archivo temporal en lieu del JSON de producción (no toca backup).",
    )
    args = parser.parse_args()

    model_p = Path(args.model_path)
    if not model_p.exists():
        log.error("Modelo no encontrado en %s", model_p)
        return 2

    artifact = joblib.load(model_p)
    model = artifact["model"]
    feature_cols = list(artifact["features"])
    trained_at = artifact.get("trained_at")
    train_seasons = list(artifact.get("train_seasons", []))
    val_seasons = list(artifact.get("val_seasons", []))
    log.info(
        "Modelo cargado: version=%s trained_at=%s train_seasons=%s val_seasons=%s n_features=%d",
        artifact.get("version"), trained_at, train_seasons, val_seasons, len(feature_cols),
    )

    if not val_seasons:
        log.error("artifact['val_seasons'] vacío — no puedo derivar el val split.")
        return 3

    df = pd.read_parquet(args.dataset_path)
    val = df[df["season"].isin(val_seasons)].copy()
    if val.empty:
        log.error("val vacío para val_seasons=%s", val_seasons)
        return 4
    log.info("Val split: %d filas sobre %s", len(val), val_seasons)

    out_path = args.out
    if args.dry_run and not out_path:
        out_path = str(_root / "data/model/team_bias_calibration_v2.regen_dryrun.json")
        log.info("dry-run → out_path=%s", out_path)

    # En dry-run: no queremos backup accidental. El helper solo crea backup si
    # out_path EXISTE Y backup NO EXISTE. Passando un backup inexistente temp
    # path asegura que no toca el backup de producción.
    backup_path = args.backup
    if args.dry_run and not backup_path:
        backup_path = str(_root / "data/model/_regen_dryrun_backup.json")

    payload = _regenerate_team_bias_calibration(
        model=model,
        val_df=val,
        feature_cols=feature_cols,
        model_trained_at=trained_at,
        model_train_seasons=train_seasons,
        out_path=out_path,
        backup_path=backup_path,
    )
    n_teams = len(payload.get("corrections", {}))
    log.info("Regeneración completa: n_teams=%d", n_teams)
    return 0


if __name__ == "__main__":
    sys.exit(main())
