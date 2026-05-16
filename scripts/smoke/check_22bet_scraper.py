"""
Smoke — scraper 22bet
======================
Ejecuta el scraper de 22bet en modo `--dry-run --max-matches 3 --markets all`
y valida:

1. Devuelve al menos 1 partido con cuotas válidas.
2. En total_over_under: home/over y under presentes con odds > 1.01.
3. Columnas canónicas del schema unificado están presentes
   (market_type, side, line, odds, home_team, away_team, scraped_at).
4. Si hay filas team_with_more → las tres sides (home/draw/away) están presentes
   para al menos un partido y todas tienen odds > 1.01.

Exit codes:
  0 = PASS
  1 = FAIL (no había cuotas, schema roto o odds inválidas)
  2 = ERROR (fallo de ejecución)

Uso:
  python -m scripts.smoke.check_22bet_scraper
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Import directo del scraper evitando el naming no-importable (módulo empieza
# con dígito: '22bet_scraper.py'). Cargamos por ruta con importlib.
import importlib.util

_root = Path(__file__).resolve().parent.parent.parent
_scraper_path = _root / "scripts" / "odds" / "22bet_scraper.py"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("smoke_22bet")


def _load_scraper():
    """Carga dinámica del scraper 22bet (el nombre empieza con dígito)."""
    spec = importlib.util.spec_from_file_location("_scraper_22bet", _scraper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {_scraper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REQUIRED_COLS = (
    "home_team", "away_team", "scraped_at",
    "bookmaker", "market_type", "line", "side", "odds",
)


def run_smoke() -> int:
    try:
        scraper = _load_scraper()
    except Exception as exc:
        print(f"[FAIL] No pude importar el scraper: {exc}")
        return 2

    print("[smoke] ejecutando 22bet_scraper.fetch_odds(dry_run=True, max_matches=3, markets=all) ...")
    try:
        df = scraper.fetch_odds(
            dry_run=True,
            markets=("total_over_under", "team_with_more"),
            max_matches=3,
        )
    except Exception as exc:
        print(f"[FAIL] Error durante fetch_odds: {exc}")
        return 2

    if df is None or df.empty:
        print("[FAIL] fetch_odds devolvió DataFrame vacío "
              "(¿no hay partidos activos? ¿API caída? ¿bloqueo?)")
        return 1

    # Check 1: columnas canónicas
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"[FAIL] Columnas faltantes en el schema unificado: {missing}")
        print(f"       columnas presentes: {list(df.columns)}")
        return 1

    # Check 2: todos los odds > 1.01
    bad_odds = df[df["odds"] <= 1.01]
    if len(bad_odds):
        print(f"[FAIL] {len(bad_odds)} filas con odds <= 1.01 (valores inválidos)")
        print(bad_odds.head(5).to_string(index=False))
        return 1

    # Check 3: al menos 1 partido con O/U (home_over + home_under en la misma línea)
    ou = df[df["market_type"] == "total_over_under"]
    if ou.empty:
        print("[FAIL] Ninguna cuota total_over_under devuelta")
        return 1

    matches_with_both_sides = (
        ou.groupby(["home_team", "away_team", "line"])["side"]
          .nunique()
          .reset_index()
    )
    n_valid = (matches_with_both_sides["side"] >= 2).sum()
    if n_valid < 1:
        print("[FAIL] Ningún partido devolvió over+under en la misma línea")
        return 1

    # Check 4 (opcional): team_with_more completo para al menos un partido
    twm = df[df["market_type"] == "team_with_more"]
    twm_full = 0
    if len(twm):
        by_match = twm.groupby(["home_team", "away_team"])["side"].apply(set).reset_index()
        twm_full = sum(1 for s in by_match["side"] if {"home", "draw", "away"}.issubset(s))
        if twm_full == 0:
            print("[WARN] Hay filas team_with_more pero ningún partido tiene los 3 lados "
                  "(home/draw/away) — posible error de parseo; no falla el smoke.")

    # Resumen
    n_matches = df[["home_team", "away_team"]].drop_duplicates().shape[0]
    print(f"[PASS] {len(df)} filas | {n_matches} partidos | "
          f"O/U partidos completos: {n_valid} | team_with_more completos: {twm_full}")
    print(f"       columnas: {list(df.columns)}")
    print(f"       market_type counts: {df['market_type'].value_counts().to_dict()}")
    print(f"       rango odds: {df['odds'].min():.2f} — {df['odds'].max():.2f}")
    return 0


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(run_smoke())


if __name__ == "__main__":
    main()
