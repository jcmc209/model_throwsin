# Throw-In Predictor — LaLiga

Modelo predictivo para el número de saques de banda por equipo en partidos de LaLiga. Diseñado para evaluación de valor en mercados de apuestas (Codere).

## Resultados actuales

| Modelo | MAE val 2024/25 |
|---|---|
| Baseline (media histórica por equipo) | 4.77 |
| **LightGBM Poisson + decay por temporada** | **4.37** |
| NegBinom (sanity check) | 4.65 |

El modelo bate el baseline en validación walk-forward (temporadas 2021/22–2023/24 → entrenamiento, 2024/25 → validación, 2025/26 → test).

## Estructura del proyecto

```
model_throwins/
├── model/                          # Pipeline de modelado
│   ├── features.py                 # Feature engineering (rolling, EWMA, opponent, etc.)
│   ├── dataset_builder.py          # Genera data/model/dataset.parquet
│   ├── train.py                    # Entrena LightGBM Poisson + NegBinom sanity check
│   └── predict.py                  # Predicciones para partidos futuros
│
├── scripts/                        # Scripts de recolección de datos
│   ├── whoscored_scraper.py        # Scraper principal de WhoScored
│   ├── weather_fetcher.py          # Descarga meteorología de Open-Meteo
│   └── run_all_seasons.py          # Ejecuta el scraper para todas las temporadas
│
├── notebooks/
│   └── 01_eda.ipynb                # EDA + análisis de feature importance + residuos
│
├── data/
│   ├── whoscored_laliga/           # Datos scrapeados por temporada (no en repo)
│   │   └── {season}/               # team_stats, throw_ins, pass_map, heatmap, all_events
│   ├── reference/                  # Datos estáticos de referencia
│   │   ├── stadiums.csv            # Dimensiones de campo + coordenadas + whoscored_id
│   │   └── liga_calendar_rows.csv  # Calendario 2023/24–2025/26 con horarios
│   └── model/                      # Artefactos del modelo (no en repo, regenerables)
│       ├── dataset.parquet
│       ├── model_v1.joblib
│       ├── metrics_v1.json         # Métricas de validación (sí en repo)
│       ├── feature_importance.csv  # Ranking de features (sí en repo)
│       └── predictions_*.parquet
│
├── requirements.txt
├── .env.example
└── openspec/                       # Spec-Driven Development (diseño + historial)
    ├── config.yaml
    ├── specs/                      # Especificaciones activas (source of truth)
    └── changes/archive/            # Historial de cambios implementados
```

## Instalación

```bash
git clone https://github.com/tu-usuario/model_throwins.git
cd model_throwins

pip install -r requirements.txt
playwright install chromium     # solo necesario para el scraper
```

Copia `.env.example` a `.env` y rellena si usas MongoDB Atlas (opcional):

```bash
cp .env.example .env
```

## Uso del pipeline de modelado

El pipeline completo, una vez que tienes los datos en `data/whoscored_laliga/`:

```bash
# 1. Descarga meteorología para todos los partidos
python scripts/weather_fetcher.py

# 2. Construye el dataset con todas las features
python -m model.dataset_builder

# 3. Entrena el modelo (LightGBM + NegBinom sanity)
python -m model.train

# 4. Predice la próxima jornada
python -m model.predict --matchday next

# O para una fecha concreta
python -m model.predict --date 2026-05-01

# O todos los partidos scheduled del calendario
python -m model.predict --all-scheduled
```

Las predicciones se guardan en `data/model/predictions_YYYYMMDD.parquet`.

## Recolección de datos

El scraper descarga datos de WhoScored para LaLiga. Requiere una sesión autenticada.

```bash
# Scraper de una sola temporada
python scripts/whoscored_scraper.py

# Todas las temporadas en secuencia
python scripts/run_all_seasons.py
```

## Features del modelo

El modelo genera ~96 features por fila (una fila = un equipo en un partido):

- **Rolling 3/5/10** — media de los N partidos previos del equipo y del rival
- **EWMA α=0.3/0.5** — media con decay exponencial (pondera más lo reciente)
- **Season-to-date** — media acumulada en la temporada actual
- **Contexto** — `is_home`, `matchday_number`, `days_since_last_match`, `has_full_history`
- **Estadio** — `pitch_length_m`, `pitch_width_m`, `capacity`
- **Meteorología** — `temperature_2m`, `wind_speed_10m`, `precipitation`, `relative_humidity_2m`, `weather_code`

Variables base: `throw_ins_total`, `corners_total`, `possession_pct`, `passes_total`, `fouls_committed`, `aerials_total`, `touches_total`.

## Datos externos gratuitos

- **WhoScored** — estadísticas de partido (scraper incluido)
- **Open-Meteo** — meteorología histórica y previsiones (sin API key, gratis)
- **Liga calendar** — calendario oficial de LaLiga (incluido en `data/reference/`)

## Diseño técnico

El proyecto usa [Spec-Driven Development](openspec/config.yaml). Las especificaciones activas están en `openspec/specs/` y el historial completo de cambios (propuestas, diseño, verificación) en `openspec/changes/archive/`.

## Próximos pasos

- [ ] `betting-evaluator` — colección de cuotas Codere + cálculo de edge del modelo
- [ ] Features de zona de saque (requiere `throw_ins.parquet` evento a evento)
- [ ] Hyperparameter tuning bayesiano
