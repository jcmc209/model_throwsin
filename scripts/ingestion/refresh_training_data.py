"""
Refresh Training Data Orchestrator
==================================
Encadena la pipeline completa de ingesta para refrescar el training dataset:

  1. whoscored scrape (via run_all_seasons)  — PLAYWRIGHT, fragile
  2. event aggregator                         — pandas puro, fast
  3. referee extractor                        — parse JSON local, fast
  4. weather fetcher                          — Open-Meteo API, fast
  5. dataset builder                          — joins + features, fast
  6. dataset builder match-level (opcional)   — fast

Uso:
  python scripts/ingestion/refresh_training_data.py
  python scripts/ingestion/refresh_training_data.py --skip-whoscored
  python scripts/ingestion/refresh_training_data.py --only aggregator,weather,dataset
  python scripts/ingestion/refresh_training_data.py --season 2025/2026
  python scripts/ingestion/refresh_training_data.py --no-match-level
  python scripts/ingestion/refresh_training_data.py --verbose
  python scripts/ingestion/refresh_training_data.py --dry-run

Comportamiento:
  - Cada paso corre como subprocess aislado (falla de uno NO mata los demás salvo si
    el siguiente lo requiere — p. ej. si whoscored falla pero ya hay data local, el
    aggregator igual corre sobre lo existente).
  - Summary final con status por paso (OK / FAIL / SKIP) + tiempo por paso.
  - Exit 0 si TODOS los pasos esenciales OK; exit 1 si falla un paso que bloquea
    pasos posteriores.

Constraint: este orquestador NO reentrena el modelo. `model.train` queda separado.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "automation"

# Status constants
STATUS_OK = "OK"
STATUS_FAIL = "FAIL"
STATUS_SKIP = "SKIP"
STATUS_TIMEOUT = "TIMEOUT"


def _build_whoscored_cmd(season: Optional[str]) -> list[str]:
    """
    Si se pasa --season, llamamos directamente al scraper con --season
    (run_all_seasons.py itera una lista hardcoded de temporadas).
    Si no, usamos run_all_seasons.py con --resume para cubrir todo.
    """
    if season:
        return [
            sys.executable,
            "scripts/ingestion/whoscored_scraper.py",
            "--season", season,
            # resume es default en el scraper (no-resume es el flag opt-in)
        ]
    return [
        sys.executable,
        "scripts/ingestion/run_all_seasons.py",
        # --resume es default True via sys.argv scan en run_all_seasons
    ]


def build_steps(season: Optional[str]) -> list[dict]:
    """Construye la lista de pasos. Season-aware para el step de whoscored."""
    return [
        {
            "name": "whoscored",
            "cmd": _build_whoscored_cmd(season),
            "timeout": 1800,   # 30 min cap
            "required_for": ["event_aggregator", "referee", "dataset_builder"],
            "network": True,
            "fragile": True,
        },
        {
            "name": "event_aggregator",
            "cmd": [sys.executable, "scripts/ingestion/event_aggregator.py"],
            "timeout": 600,
            "required_for": ["dataset_builder"],
            "network": False,
            "fragile": False,
        },
        {
            "name": "referee",
            "cmd": [sys.executable, "scripts/ingestion/referee_extractor.py"],
            "timeout": 300,
            "required_for": ["dataset_builder"],
            "network": False,
            "fragile": False,
        },
        {
            "name": "weather",
            "cmd": [sys.executable, "scripts/ingestion/weather_fetcher.py"],
            "timeout": 900,
            "required_for": ["dataset_builder"],
            "network": True,
            "fragile": False,
        },
        {
            "name": "dataset_builder",
            "cmd": [sys.executable, "-m", "model.dataset_builder"],
            "timeout": 600,
            "required_for": ["dataset_builder_match"],
            "network": False,
            "fragile": False,
        },
        {
            "name": "dataset_builder_match",
            "cmd": [sys.executable, "-m", "model.dataset_builder_match"],
            "timeout": 600,
            "required_for": [],
            "network": False,
            "fragile": False,
            "optional": True,
        },
    ]


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

def setup_logging(verbose: bool) -> Path:
    """Configura logging a stdout + archivo. Devuelve el path del log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"refresh_{ts}.log"

    level = logging.DEBUG if verbose else logging.INFO

    # Limpiar handlers previos para idempotencia si se importa
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


log = logging.getLogger("refresh_training_data")


# ─────────────────────────────────────────────────────────────
# STEP RUNNER
# ─────────────────────────────────────────────────────────────

def run_step(step: dict, verbose: bool) -> dict:
    """
    Corre un step como subprocess. Devuelve:
      {status, duration_s, stdout_tail, stderr_tail, returncode}
    """
    name = step["name"]
    cmd = step["cmd"]
    timeout = step["timeout"]

    log.info("[START] %s — cmd=%s (timeout=%ds)", name, " ".join(cmd), timeout)
    t0 = time.monotonic()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - t0
        log.error("[TIMEOUT] %s — se agotó el timeout de %ds", name, timeout)
        stdout_tail = (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else ""
        stderr_tail = (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else ""
        return {
            "status": STATUS_TIMEOUT,
            "duration_s": round(duration, 1),
            "returncode": None,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    except Exception as exc:
        duration = time.monotonic() - t0
        log.exception("[FAIL] %s — excepción al lanzar subprocess: %s", name, exc)
        return {
            "status": STATUS_FAIL,
            "duration_s": round(duration, 1),
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }

    duration = time.monotonic() - t0
    stdout_tail = (proc.stdout or "")[-2000:]
    stderr_tail = (proc.stderr or "")[-2000:]

    if verbose:
        if proc.stdout:
            log.debug("[%s stdout]\n%s", name, proc.stdout)
        if proc.stderr:
            log.debug("[%s stderr]\n%s", name, proc.stderr)

    if proc.returncode == 0:
        log.info("[OK] %s — %.1fs", name, duration)
        return {
            "status": STATUS_OK,
            "duration_s": round(duration, 1),
            "returncode": proc.returncode,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }

    log.error(
        "[FAIL] %s — returncode=%d (%.1fs). stderr tail:\n%s",
        name, proc.returncode, duration, stderr_tail or "(empty)",
    )
    return {
        "status": STATUS_FAIL,
        "duration_s": round(duration, 1),
        "returncode": proc.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


# ─────────────────────────────────────────────────────────────
# STEP FILTERING
# ─────────────────────────────────────────────────────────────

def filter_steps(
    steps: list[dict],
    only: Optional[list[str]],
    skip_whoscored: bool,
    no_match_level: bool,
) -> list[dict]:
    """Aplica los CLI flags de filtering."""
    out = []
    for s in steps:
        if only is not None and s["name"] not in only:
            continue
        if skip_whoscored and s["name"] == "whoscored":
            continue
        if no_match_level and s["name"] == "dataset_builder_match":
            continue
        out.append(s)
    return out


def validate_only_list(only: list[str], all_steps: list[dict]) -> None:
    """Verifica que todos los nombres en --only existen."""
    valid = {s["name"] for s in all_steps}
    invalid = [x for x in only if x not in valid]
    if invalid:
        raise SystemExit(
            f"--only contiene pasos inválidos: {invalid}. "
            f"Válidos: {sorted(valid)}"
        )


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

def print_summary(results: dict[str, dict], total_elapsed: float, log_path: Path) -> int:
    """Imprime tabla de resumen. Devuelve el exit code."""
    log.info("=" * 70)
    log.info("SUMMARY — refresh_training_data")
    log.info("=" * 70)
    header = f"{'step':<24} {'status':<10} {'duration_s':>10}  rc"
    log.info(header)
    log.info("-" * 70)

    any_essential_fail = False
    for name, r in results.items():
        rc = r.get("returncode")
        rc_str = str(rc) if rc is not None else "-"
        log.info("%-24s %-10s %10.1f  %s", name, r["status"], r["duration_s"], rc_str)
        if r["status"] not in (STATUS_OK, STATUS_SKIP) and r.get("essential", True):
            # whoscored failure is NOT essential-fatal: pipeline downstream
            # can still run on existing local data. We only treat as fatal
            # if a REQUIRED-for-others step failed AND later steps couldn't run.
            pass

    log.info("-" * 70)
    log.info("Total wall-clock: %.1fs (%.2f min)", total_elapsed, total_elapsed / 60.0)
    log.info("Log file: %s", log_path)
    log.info("=" * 70)

    # Exit code: 0 si TODOS los pasos essential non-optional terminaron OK o SKIP
    # por dependencia (pero dataset_builder SÍ debe ser OK para considerar éxito).
    # 1 si dataset_builder falló (el entregable principal).
    dsb = results.get("dataset_builder")
    if dsb is not None and dsb["status"] == STATUS_FAIL:
        return 1
    # Si dsb fue skipped por dependencia fallida catastrófica, también es fail.
    if dsb is not None and dsb["status"] == STATUS_SKIP and dsb.get("skip_reason", "").startswith("dependency"):
        return 1
    # Cualquier FAIL en steps non-optional que no sea whoscored (fragile) también cuenta.
    for name, r in results.items():
        if r["status"] == STATUS_FAIL and name not in ("whoscored", "dataset_builder_match"):
            return 1
    return 0


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh training dataset runing the full ingestion pipeline.",
    )
    parser.add_argument(
        "--skip-whoscored", action="store_true",
        help="Skip step 1 (útil cuando Whoscored está roto y querés rebuild-ear features de data local).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated list de steps a correr (ej: event_aggregator,dataset_builder).",
    )
    parser.add_argument(
        "--season", default=None,
        help="Si se provee (ej: 2025/2026), el step de whoscored corre sólo esa temporada.",
    )
    parser.add_argument(
        "--no-match-level", action="store_true",
        help="Skip step 6 (dataset_builder_match).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="DEBUG logs + stdout/stderr completo de cada subprocess.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Lista los comandos a correr sin ejecutar.",
    )
    args = parser.parse_args()

    log_path = setup_logging(args.verbose)
    log.info("=" * 70)
    log.info("refresh_training_data — start %s", datetime.now().isoformat(timespec="seconds"))
    log.info("project_root: %s", PROJECT_ROOT)
    log.info("args: %s", vars(args))
    log.info("=" * 70)

    all_steps = build_steps(args.season)

    only_list: Optional[list[str]] = None
    if args.only:
        only_list = [x.strip() for x in args.only.split(",") if x.strip()]
        validate_only_list(only_list, all_steps)

    planned = filter_steps(
        all_steps,
        only=only_list,
        skip_whoscored=args.skip_whoscored,
        no_match_level=args.no_match_level,
    )

    if not planned:
        log.warning("Ningún step seleccionado. Salgo.")
        return 0

    log.info("Steps planeados: %s", [s["name"] for s in planned])

    if args.dry_run:
        log.info("--dry-run activo. Comandos:")
        for s in planned:
            log.info("  [%s] %s", s["name"], " ".join(s["cmd"]))
        return 0

    # Track cuales steps fallan para marcar dependientes como SKIP.
    results: dict[str, dict] = {}
    failed_names: set[str] = set()

    t_total = time.monotonic()

    for step in planned:
        name = step["name"]

        # Si algún step previo requerido falló, marcar como SKIP (dependency failed)
        failed_deps = [
            f for f in failed_names
            if name in next((s["required_for"] for s in all_steps if s["name"] == f), [])
        ]
        if failed_deps:
            log.warning(
                "[SKIP] %s — dependencia(s) fallida(s): %s", name, failed_deps,
            )
            results[name] = {
                "status": STATUS_SKIP,
                "duration_s": 0.0,
                "returncode": None,
                "stdout_tail": "",
                "stderr_tail": "",
                "skip_reason": f"dependency failed: {failed_deps}",
            }
            continue

        r = run_step(step, verbose=args.verbose)
        r["essential"] = not step.get("optional", False)
        results[name] = r

        if r["status"] in (STATUS_FAIL, STATUS_TIMEOUT):
            # Whoscored falla = no bloquea aggregator/referee/dataset_builder
            # (corren sobre la data local existente). Sólo loguear WARN.
            if name == "whoscored":
                log.warning(
                    "[WARN] whoscored falló (%s) — los siguientes steps igual corren "
                    "sobre la data local existente.", r["status"],
                )
            else:
                failed_names.add(name)

    total_elapsed = time.monotonic() - t_total
    exit_code = print_summary(results, total_elapsed, log_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
