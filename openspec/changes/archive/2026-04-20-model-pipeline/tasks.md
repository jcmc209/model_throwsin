# Tasks: model-pipeline

## Phase 1: Foundation

- [x] 1.1 Crear directorios `model/` y `data/model/` (con `.gitkeep`)
- [x] 1.2 Crear `model/__init__.py` vacío
- [x] 1.3 Actualizar `requirements.txt` con `lightgbm`, `scikit-learn`, `statsmodels`, `joblib`, `jupyter`
- [x] 1.4 Instalar dependencias (`pip install -r requirements.txt`) y verificar imports
- [x] 1.5 Añadir referencia a `model/` en `openspec/config.yaml` (sección `context`)

## Phase 2: Feature engineering (`model/features.py`)

- [x] 2.1 Crear `model/features.py` con logging estilo repo (handler a `model_training.log`)
- [x] 2.2 Implementar `compute_rolling(df, group_col, target_cols, windows=[3,5,10])` con shift(1) para evitar leakage
- [x] 2.3 Implementar `compute_ewma(df, group_col, target_cols, alphas=[0.3, 0.5])` con shift(1)
- [x] 2.4 Implementar `compute_opponent_features(df)` que una rolling/EWMA del rival por `match_id` + `is_home` invertido
- [x] 2.5 Implementar `compute_season_to_date(df, target_cols)` — media acumulada por equipo y temporada
- [x] 2.6 Implementar `impute_weather_forecast_gap(df)` — media por estadio × mes para partidos >16d vista

## Phase 3: Dataset builder (`model/dataset_builder.py`)

- [x] 3.1 Crear `model/dataset_builder.py` con `CONFIG` dict, logging, `__main__` con `argparse`
- [x] 3.2 Cargar todos los `team_stats.parquet` de `data/whoscored_laliga/**/` y concatenar
- [x] 3.3 Construir vista larga `(match_id, team_id, is_home)` con target `throw_ins_total`
- [x] 3.4 Aplicar `compute_rolling`, `compute_ewma`, `compute_season_to_date` sobre `throw_ins, corners, possession, passes, fouls, aerials, touches`
- [x] 3.5 Aplicar `compute_opponent_features` para features del rival
- [x] 3.6 Añadir contexto: `matchday_number`, `days_since_last_match`, `has_full_history`
- [x] 3.7 Join con `stadiums.csv` (por `team_id` local) y con `weather.parquet` (por `match_id`)
- [x] 3.8 Assert 3.622 filas, target sin nulls, features obligatorias sin nulls
- [x] 3.9 Guardar `data/model/dataset.parquet` y loguear shape + resumen de columnas

## Phase 4: Training (`model/train.py`)

- [x] 4.1 Crear `model/train.py` con logging, `CONFIG`, `argparse` (`--weights uniform|decay|both`)
- [x] 4.2 Cargar `dataset.parquet`, aplicar split temporal (train 21-24 / val 24-25 / test 25-26), assert sin intersección
- [x] 4.3 Calcular baseline MAE (media por equipo) en validación y loguearlo
- [x] 4.4 Entrenar LightGBM Poisson con early_stopping_rounds=50
- [x] 4.5 Entrenar segundo LightGBM con `sample_weight` decay por temporada (0.6/0.8/1.0)
- [x] 4.6 Entrenar NegBinom (statsmodels) como sanity check sobre features reducidas
- [x] 4.7 Reportar MAE, RMSE, MAE por equipo, MAE por temporada de los 3 modelos en val
- [x] 4.8 Seleccionar mejor modelo por val MAE, warning si no bate 4.84
- [x] 4.9 Guardar `model_v1.joblib` (dict con modelo + features + params + val_mae + weights_scheme)
- [x] 4.10 Guardar `metrics_v1.json` y `feature_importance.csv`

## Phase 5: Prediction (`model/predict.py`)

- [x] 5.1 Crear `model/predict.py` con `argparse` (`--matchday next` | `--date YYYY-MM-DD` | `--all-scheduled`)
- [x] 5.2 Abortar con mensaje claro si falta `model_v1.joblib`
- [x] 5.3 Cargar calendario, filtrar status `scheduled` según argumento
- [x] 5.4 Construir filas de inferencia usando `model/features.py` (solo partidos con fecha < F)
- [x] 5.5 Aplicar `impute_weather_forecast_gap` para partidos >16d vista
- [x] 5.6 Validar que columnas e orden coinciden con `model["features"]`, lanzar error si no
- [x] 5.7 Generar predicciones home/away y `pred_throw_ins_total = home + away`
- [x] 5.8 Guardar `predictions_YYYYMMDD.parquet` e imprimir tabla resumida

## Phase 6: Notebook y verificación

- [x] 6.1 Crear `notebooks/01_eda.ipynb` con EDA del dataset (distribuciones, correlaciones)
- [x] 6.2 Añadir celdas post-entreno: feature importance, ablation top-N, residuos por equipo
- [x] 6.3 Verificar anti-leakage: para 3 filas random, inspeccionar que rolling solo usa fechas < F
- [x] 6.4 Verificar reproducibilidad: 2 ejecuciones de `dataset_builder.py` dan parquet idéntico (`assert_frame_equal`)
- [x] 6.5 Smoke-test `predict.py` con próxima jornada scheduled y revisar salida
