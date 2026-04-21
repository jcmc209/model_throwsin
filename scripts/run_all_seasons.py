"""
Lanza el scraper para las últimas 4 temporadas completas de LaLiga.
Ejecutar desde la raíz del proyecto con:

    python scripts/run_all_seasons.py

Opciones:
    --no-headless   Mostrar el navegador (útil para depurar)
    --no-mongo      Desactivar MongoDB (solo guarda en local)
    --format        csv / parquet / both (default: both)
    --resume        Reanudar desde donde se dejó (default: True)
"""
import sys
import logging
from scripts.whoscored_scraper import run_scraper, CONFIG, setup_logging

setup_logging("scraper_all_seasons.log")
log = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────
SEASONS = [
    "2025/2026",   # temporada actual (partidos jugados hasta hoy)
    "2024/2025",
    "2023/2024",
    "2022/2023",
    "2021/2022",
    "2020/2021",   # nueva
    "2019/2020",   # nueva
]

USE_MONGO   = "--no-mongo"    not in sys.argv
HEADLESS    = "--no-headless" not in sys.argv
RESUME      = "--no-resume"   not in sys.argv
OUTPUT_FMT  = "both"
for arg in sys.argv:
    if arg.startswith("--format="):
        OUTPUT_FMT = arg.split("=")[1]

CONFIG["headless"]      = HEADLESS
CONFIG["output_format"] = OUTPUT_FMT

# ── Estimación de tiempo ───────────────────────────────────
# ~380 partidos/temporada × 4 temporadas = ~1520 partidos
# ~6 seg/partido (delay mín) → ~2.5 horas mínimo
# En práctica con reintentos y carga de página: 4-8 horas
log.info("=" * 60)
log.info("WhoScored LaLiga — Scraping 4 temporadas")
log.info(f"Temporadas: {', '.join(SEASONS)}")
log.info(f"MongoDB: {'activado' if USE_MONGO else 'desactivado'}")
log.info(f"Headless: {HEADLESS}")
log.info(f"Formato salida: {OUTPUT_FMT}")
log.info("Estimacion: 8-16 horas (~380 partidos x 6 completas + partidos jugados en 2025/2026)")
log.info("Puedes cerrar y reanudar: el script continua donde lo dejo")
log.info("=" * 60)

# ── Loop principal ─────────────────────────────────────────
for i, season in enumerate(SEASONS, 1):
    log.info(f"\n{'='*60}")
    log.info(f"[{i}/{len(SEASONS)}] Iniciando temporada: {season}")
    log.info(f"{'='*60}")

    try:
        run_scraper(
            season_key         = season,
            match_ids_override = None,
            resume             = RESUME,
            consolidate        = True,
            use_mongo          = USE_MONGO,
        )
        log.info(f"[{i}/{len(SEASONS)}] Temporada {season} completada OK")

    except KeyboardInterrupt:
        log.warning(f"\nInterrumpido por el usuario en temporada {season}.")
        log.warning("Progreso guardado. Ejecuta de nuevo para continuar.")
        sys.exit(0)

    except Exception as e:
        log.error(f"[{i}/{len(SEASONS)}] Error en temporada {season}: {e}", exc_info=True)
        log.warning("Continuando con la siguiente temporada...")

log.info("\n" + "=" * 60)
log.info("SCRAPING COMPLETADO — todas las temporadas procesadas")
log.info("Datos en: data/whoscored_laliga/")
log.info("=" * 60)
