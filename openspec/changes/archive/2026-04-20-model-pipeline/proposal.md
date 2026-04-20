# Proposal: model-pipeline

## Intent

Construir el pipeline de modelado predictivo para el número de saques de banda por equipo en partidos de LaLiga. El pipeline debe ser reproducible, batir el baseline (MAE 4.84) y servir como base para la evaluación posterior contra cuotas de mercado.

## Scope

### In Scope
- `model/dataset_builder.py` — genera dataset ancho (~40 features) desde `team_stats` + `stadiums.csv` + `weather.parquet`
- `model/train.py` — LightGBM Poisson + NegBinom sanity check; evaluación con MAE/RMSE; serialización
- `model/predict.py` — inferencia para partidos futuros (calendario scheduled)
- `notebooks/01_eda.ipynb` — exploración inicial y análisis de feature importance
- Split temporal: train 2021-2024, val 2024/25, test 2025/26
- Output del dataset: `data/model/dataset.parquet`
- Modelo serializado: `data/model/model_v1.pkl`

### Out of Scope
- Integración con cuotas de mercado (cambio posterior `betting-evaluator`)
- Features de zona/ubicación de saques (requiere usar `throw_ins.parquet` — iteración futura)
- Ensembles multi-modelo o stacking
- Hyperparameter tuning exhaustivo (bayesiano, etc.) — solo grid search manual básico

## Approach

**Híbrido** (notebook + scripts). Modelo conjunto (home y away en una sola tabla con `is_home` como feature).

1. `dataset_builder.py` calcula rolling (3, 5, 10) para `throw_ins`, `corners`, `possession`, `passes`, `fouls`, `aerials`, `touches` — del equipo y del oponente. Añade season-to-date averages, contexto de partido, estadio y meteorología.
2. `train.py` entrena LightGBM Poisson con validación 2024/25, hace feature importance + ablation, guarda el modelo final.
3. `predict.py` toma partidos futuros, construye sus features (rolling hasta la fecha de predicción), devuelve predicción por equipo.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `model/` | New | Directorio con los 3 scripts del pipeline |
| `notebooks/01_eda.ipynb` | New | EDA + análisis post-entrenamiento |
| `data/model/` | New | Dataset y modelo serializados |
| `requirements.txt` | Modified | +`lightgbm`, +`scikit-learn`, +`statsmodels`, +`jupyter` |
| `openspec/config.yaml` | Modified | +referencia a `model/` |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Overfit con 1.139 filas train + 40 features | Med | Early stopping, regularización LightGBM, ablation |
| Equipos ascendidos sin historial rolling | Low | Fallback a media global en primeros 5 partidos |
| No batir baseline | Med | Si el modelo no mejora MAE 4.84, iterar features antes de cerrar |
| No estacionariedad entre temporadas | Low | Walk-forward natural; evaluar por temporada |

## Rollback Plan

- Todo vive en `model/` y `data/model/` — directorios nuevos, borrar los elimina sin afectar al resto del proyecto.
- Los datos fuente (`team_stats`, `weather`, `stadiums`) no se modifican — siguen intactos.

## Dependencies

- `lightgbm`, `scikit-learn`, `statsmodels`, `jupyter` (nuevas)
- `pandas`, `numpy` (ya presentes)
- `data/whoscored_laliga/**/team_stats.parquet`, `data/reference/weather.parquet`, `data/reference/stadiums.csv` (ya disponibles)

## Success Criteria

- [ ] `dataset.parquet` contiene 1.811 filas × 2 (home+away) con ~40 features y sin leakage
- [ ] Modelo entrenado bate el baseline de media por equipo (MAE < 4.84) en validación 2024/25
- [ ] `predict.py` devuelve predicciones coherentes para partidos scheduled del calendario
- [ ] Feature importance y ablation documentados en el notebook
- [ ] Todo reproducible con `python model/dataset_builder.py && python model/train.py`
