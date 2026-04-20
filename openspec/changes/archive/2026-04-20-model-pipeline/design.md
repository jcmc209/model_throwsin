# Design: model-pipeline

## Technical Approach

Tres scripts Python independientes + un notebook de EDA, orquestados por CLI. Cada script es idempotente, lee de disco, escribe a disco. Sin servicio, sin estado compartido. Sigue el patrón ya establecido por `whoscored_scraper.py` y `weather_fetcher.py`: logging a archivo, `CONFIG` dict al principio, `__main__` con `argparse`.

El modelo es un único LightGBM Poisson entrenado sobre filas `(match_id, is_home)`. Las features rolling + EWMA se calculan offline en `dataset_builder.py` y se persisten en `dataset.parquet`, así `train.py` solo lee y entrena.

## Architecture Decisions

| Decisión | Opciones | Elección | Rationale |
|---|---|---|---|
| Estructura | 1 notebook todo-en-uno / 3 scripts + notebook | **3 scripts + notebook** | Reproducibilidad CLI, notebook solo para EDA/análisis post-entreno. Coincide con el patrón del repo. |
| Modelo | 2 modelos (home/away) / 1 con `is_home` | **1 con `is_home`** | +100% de datos por modelo, captura interacción home-away vía features del oponente. |
| Rolling | ventanas fijas / solo EWMA / ambas | **rolling 3/5/10 + EWMA α=0.3/0.5** | Rolling para referencia estable, EWMA para pondear recencia. El modelo elige via `feature_importance`. |
| Persistencia modelo | pickle / joblib / LGBM nativo | **joblib** | Serializa `dict` con modelo + metadata (versión, features, params) en un solo fichero. |
| Formato dataset | CSV / Parquet | **Parquet** | Tipos preservados, 5-10× menor, ya es el estándar del proyecto. |
| Imputación weather futuro | NaN / media estadio-mes | **media estadio-mes** | Evita que LightGBM use NaN como señal espuria en partidos lejanos. |

## Data Flow

```
team_stats.parquet ─┐
stadiums.csv ───────┼─→ dataset_builder.py ──→ data/model/dataset.parquet
weather.parquet ────┘                              │
                                                   ▼
                              train.py ──→ data/model/model_v1.joblib
                                   │        data/model/metrics_v1.json
                                   │        data/model/feature_importance.csv
                                   ▼
                          notebook 01_eda.ipynb
                                   
liga_calendar_rows.csv ─→ predict.py ──→ data/model/predictions_YYYYMMDD.parquet
(scheduled)    +  model_v1.joblib
```

## File Changes

| File | Action | Description |
|---|---|---|
| `model/__init__.py` | Create | Marcar paquete |
| `model/dataset_builder.py` | Create | Genera `dataset.parquet` con rolling + EWMA + weather + stadium |
| `model/features.py` | Create | Funciones puras: `compute_rolling`, `compute_ewma`, `compute_opponent_features` (reutilizadas por `dataset_builder` y `predict`) |
| `model/train.py` | Create | Entrena LightGBM Poisson, NegBinom sanity check, evalúa, serializa |
| `model/predict.py` | Create | Carga modelo, construye features para partidos scheduled, predice |
| `notebooks/01_eda.ipynb` | Create | EDA, análisis de feature importance post-entreno |
| `data/model/.gitkeep` | Create | Directorio persistente |
| `requirements.txt` | Modify | +lightgbm, +scikit-learn, +statsmodels, +joblib, +jupyter |
| `openspec/config.yaml` | Modify | Añadir referencia a `model/` |

## Interfaces / Contracts

### `dataset.parquet` — esquema

```python
{
    "match_id": int,            # PK junto con is_home
    "team_id": int,
    "opponent_id": int,
    "is_home": int,             # 0|1
    "match_date": datetime,
    "season": str,              # "2023-2024"
    "matchday_number": int,
    "days_since_last_match": int,
    "has_full_history": bool,
    # Rolling team: rolling{3,5,10}_{throw_ins,corners,possession,passes,fouls,aerials,touches}
    # Rolling opp:  opp_rolling{3,5,10}_{...}
    # EWMA team:    ewma_alpha{03,05}_{...}
    # EWMA opp:     opp_ewma_alpha{03,05}_{...}
    # Season-to-date: std_{throw_ins,corners,possession}
    "pitch_length_m": int, "pitch_width_m": int, "capacity": int,
    "temperature_2m": float, "wind_speed_10m": float,
    "precipitation": float, "relative_humidity_2m": float, "weather_code": int,
    "weather_imputed": bool,
    "throw_ins_total": int,     # TARGET
}
```

### `model_v1.joblib` — estructura

```python
{
    "model": LGBMRegressor,
    "version": "v1",
    "trained_at": "2026-04-20T12:00:00",
    "features": ["rolling5_throw_ins", ...],   # orden exacto esperado en predict
    "params": {...},
    "val_mae": 4.12,
    "sample_weights_scheme": "uniform" | "decay",
}
```

### CLI

```bash
python -m model.dataset_builder              # genera dataset.parquet
python -m model.train                        # entrena + evalúa + guarda model_v1.joblib
python -m model.train --weights decay        # experimento con decay por temporada
python -m model.predict --matchday next      # predice próxima jornada scheduled
python -m model.predict --date 2026-05-01    # predice todos los scheduled de ese día
```

## Testing Strategy

| Layer | What | How |
|---|---|---|
| Sanity (script) | 0 nulls en features obligatorias, 3.622 filas en dataset | Asserts al final de `dataset_builder.py` |
| Anti-leakage | target no aparece en features, rolling solo usa partidos previos | Test manual en notebook: para 1 fila, verificar índices usados |
| Metric floor | MAE val < 4.84 | Chequeo en `train.py` con warning si falla |
| Reproducibilidad | Dos runs de `dataset_builder` dan el mismo hash | Comparar parquet con `pd.testing.assert_frame_equal` |

No hay infra de tests (`pytest`) en el repo. Mantenemos el estilo: asserts en scripts + verificaciones manuales en notebook.

## Migration / Rollout

No migration required. Todo vive en `model/` y `data/model/` (directorios nuevos). Los datos fuente no se tocan.

## Open Questions

- [ ] ¿`matchday_number` lo saco del calendario o lo infiero por orden de fecha? (Prefiero del calendario para consistencia con `scheduled`.)
- [ ] ¿Qué hacer si un equipo ascendido no tiene datos de rolling en 2025/26? Ya cubierto por `has_full_history`, pero ver si LGBM con NaN (sin imputar) funciona mejor que imputar con media.
- [ ] Hyperparams iniciales LightGBM: arrancamos con defaults + `num_leaves=31`, `learning_rate=0.05`, `min_child_samples=20` y tuneamos si hace falta.
