"""
Setup Task Scheduler (Windows)
==============================
Registra `odds_scheduler.py` como tarea programada de Windows que arranca
al iniciar sesión. Así el scheduler está siempre activo en segundo plano
sin tener que acordarse de abrirlo.

Uso (dos líneas separadas; no pegues comandos en la misma línea):
  python scripts/automation/setup_task_scheduler.py --remove
  python scripts/automation/setup_task_scheduler.py

Requisito: terminal **como administrador** (clic derecho → Ejecutar como
administrador). Sin eso, `schtasks` suele responder: ERROR: Access is denied.

Si ves `can't open file '...setup_task_scheduler.py~python'`, pegaste dos
comandos sin salto de línea (por ejemplo `...py~python ...`).

Comprobar que la tarea existe y está bien (desde CMD, no Git Bash sin //c):
  schtasks /Query /TN ThrowInsOddsScheduler /FO LIST /V

Forzar una ejecución ahora (comprueba en el Administrador de tareas un python.exe
y en la carpeta del proyecto `odds_scheduler.log`):
  schtasks /Run /TN ThrowInsOddsScheduler

Si registraste la tarea **antes** de usar `cmd /c cd` al proyecto, vuelve a
`--remove` y registrar: sin eso el log y las rutas `data/` pueden fallar.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TASK_NAME = "ThrowInsOddsScheduler"


def _is_windows() -> bool:
    return os.name == "nt"


def _is_admin() -> bool:
    if not _is_windows():
        return True
    try:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def register() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    python_exe = Path(sys.executable).resolve()
    scheduler_path = (project_root / "scripts" / "odds" / "odds_scheduler.py").resolve()

    # Task Scheduler no rellena «Iniciar en» con /TR solo-python → el CWD suele ser
    # System32 y rompen rutas relativas (data/, scripts/). Envolvemos en cmd /c cd.
    root_s = str(project_root)
    py_s = str(python_exe)
    sched_s = str(scheduler_path)
    task_run = (
        'cmd.exe /c "cd /d \\"'
        + root_s
        + '\\" && \\"'
        + py_s
        + '\\" \\"'
        + sched_s
        + '\\"'
        + '"'
    )

    # schtasks: ONLOGON; el comando real hace cd al proyecto antes de python
    cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/SC", "ONLOGON",
        "/TR", task_run,
        "/RL", "LIMITED",
        "/F",  # overwrite si existe
    ]
    print("Ejecutando:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or "").strip() or (r.stdout or "").strip()
        print("ERROR:", err or f"código {r.returncode}")
        if "denied" in err.lower() or "acceso" in err.lower():
            print()
            print("Abre Git Bash / PowerShell / CMD **como administrador** y vuelve a ejecutar este script.")
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
        err = (r.stderr or "").strip() or (r.stdout or "").strip()
        print("ERROR:", err or f"código {r.returncode}")
        if "denied" in err.lower():
            print("Prueba de nuevo en una terminal **como administrador**.")
        sys.exit(1)
    print("Tarea eliminada OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    if _is_windows() and not _is_admin():
        print("AVISO: No parece que esta sesión tenga privilegios de administrador.")
        print("        `schtasks` suele fallar con «Access is denied» sin ellos.")
        print("        Cierra la terminal, ábrela con «Ejecutar como administrador» y repite.")
        print()

    remove() if args.remove else register()


if __name__ == "__main__":
    main()
