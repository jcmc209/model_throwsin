"""
Lanza el scraper para las últimas 4 temporadas completas de LaLiga.
Ejecutar desde la raíz del proyecto con:

    python scripts/ingestion/run_all_seasons.py

Opciones:
    --no-headless   Mostrar el navegador (útil para depurar)
    --no-mongo      Desactivar MongoDB (solo guarda en local)
    --format        csv / parquet / both (default: both)
    --resume        Reanudar desde donde se dejó (default: True)
"""
import sys
import logging
from scripts.ingestion.whoscored_scraper import run_scraper, CONFIG, setup_logging

SEASONS = [
    "2025/2026",   # temporada actual (partidos jugados hasta hoy)
    "2024/2025",
    "2023/2024",
    "2022/2023",
    "2021/2022",
    "2020/2021",
    "2019/2020",
]


def main() -> None:
    setup_logging("scraper_all_seasons.log")
    log = logging.getLogger(__name__)

    use_mongo  = "--no-mongo"    not in sys.argv
    headless   = "--no-headless" not in sys.argv
    resume     = "--no-resume"   not in sys.argv
    output_fmt = "both"
    for arg in sys.argv:
        if arg.startswith("--format="):
            output_fmt = arg.split("=")[1]

    CONFIG["headless"]      = headless
    CONFIG["output_format"] = output_fmt

    log.info("=" * 60)
    log.info("WhoScored LaLiga — Scraping %d temporadas", len(SEASONS))
    log.info("Temporadas: %s", ", ".join(SEASONS))
    log.info("MongoDB: %s", "activado" if use_mongo else "desactivado")
    log.info("Headless: %s", headless)
    log.info("Formato salida: %s", output_fmt)
    log.info("Estimacion: 8-16 horas (~380 partidos x 6 completas + partidos jugados en 2025/2026)")
    log.info("Puedes cerrar y reanudar: el script continua donde lo dejo")
    log.info("=" * 60)

    for i, season in enumerate(SEASONS, 1):
        log.info("\n%s", "=" * 60)
        log.info("[%d/%d] Iniciando temporada: %s", i, len(SEASONS), season)
        log.info("%s", "=" * 60)

        try:
            run_scraper(
                season_key         = season,
                match_ids_override = None,
                resume             = resume,
                consolidate        = True,
                use_mongo          = use_mongo,
            )
            log.info("[%d/%d] Temporada %s completada OK", i, len(SEASONS), season)

        except KeyboardInterrupt:
            log.warning("\nInterrumpido por el usuario en temporada %s.", season)
            log.warning("Progreso guardado. Ejecuta de nuevo para continuar.")
            sys.exit(0)

        except Exception as e:
            log.error("[%d/%d] Error en temporada %s: %s", i, len(SEASONS), season, e, exc_info=True)
            log.warning("Continuando con la siguiente temporada...")

    log.info("\n%s", "=" * 60)
    log.info("SCRAPING COMPLETADO — todas las temporadas procesadas")
    log.info("Datos en: data/whoscored_laliga/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
