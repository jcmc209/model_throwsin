# Saques de banda — LaLiga

Proyecto para predecir cuántos saques de banda tiene cada equipo en partidos de LaLiga, usando datos de partido y contexto (estadio, clima, etc.).

## Estructura

- `model/` — Dataset, entrenamiento y predicción
- `scripts/ingestion/` — Recolección de datos (WhoScored, clima, árbitros, eventos)
- `scripts/odds/` — Scraping de cuotas (22bet, Codere) y scheduler
- `scripts/tuning/` — Búsqueda de hiperparámetros y validación de cambios
- `scripts/automation/` — Integración con Task Scheduler de Windows
- `data/` — Datos locales (gran parte no va al repo; ver `.gitignore`)
- `notebooks/` — Exploración
- `logs/` — Logs de ejecución (no va al repo)

## Instalación

```bash
pip install -r requirements.txt
playwright install chromium   # solo si usas el scraper con Playwright
```

## Flujo típico

Con datos ya descargados en `data/whoscored_laliga/`:

```bash
python scripts/ingestion/weather_fetcher.py
python -m model.dataset_builder
python -m model.train
python -m model.predict --matchday next
```

Para varias temporadas con el scraper:

```bash
python -m scripts.ingestion.run_all_seasons
```

Los detalles de features, calendario y referencias están en el código y en `data/reference/`.
