"""
Setup Task Scheduler (Windows)
==============================
Registra `odds_scheduler.py` como tarea programada de Windows que arranca
al iniciar sesión. Así el scheduler está siempre activo en segundo plano
sin tener que acordarse de abrirlo.

Uso:
  python scripts/setup_task_scheduler.py            # registra la tarea
  python scripts/setup_task_scheduler.py --remove   # la elimina
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TASK_NAME = "ThrowInsOddsScheduler"


def register() -> None:
    project_root = Path(__file__).resolve().parent.parent
    python_exe = sys.executable
    scheduler_path = project_root / "scripts" / "odds_scheduler.py"

    # schtasks: ONLOGON con working directory correcto
    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/SC", "ONLOGON",
        "/TR", f'"{python_exe}" "{scheduler_path}"',
        "/RL", "LIMITED",
        "/F",  # overwrite si existe
    ]
    print("Ejecutando:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR:", r.stderr)
        sys.exit(1)
    print("Tarea registrada OK:", r.stdout.strip())
    print()
    print("Para arrancarla manualmente ahora mismo:")
    print(f'  schtasks /Run /TN "{TASK_NAME}"')
    print("Para ver estado:")
    print(f'  schtasks /Query /TN "{TASK_NAME}"')


def remove() -> None:
    r = subprocess.run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR:", r.stderr)
        sys.exit(1)
    print("Tarea eliminada OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    remove() if args.remove else register()


if __name__ == "__main__":
    main()
