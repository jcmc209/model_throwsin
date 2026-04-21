# training Specification

## Purpose

Define cómo se entrena, evalúa y serializa el modelo predictivo de saques de banda. Incluye split temporal, métricas obligatorias, criterio de éxito frente al baseline y guardado del modelo.

## Requirements

### Requirement: Split temporal walk-forward

El sistema MUST dividir el dataset por temporada sin shuffle y sin fuga temporal.

#### Scenario: Partición estándar
- GIVEN el dataset con 5 temporadas
- WHEN se aplica el split
- THEN train = {2021/22, 2022/23, 2023/24}
- AND validation = {2024/25}
- AND test = {2025/26}

#### Scenario: Ninguna fila aparece en dos splits
- WHEN se construyen train/val/test
- THEN la intersección de `match_id` entre cualquier par de splits es vacía

### Requirement: Modelo principal LightGBM Poisson

El sistema MUST entrenar un modelo LightGBM con `objective='poisson'` sobre todo el conjunto de features del dataset.

#### Scenario: Entrenamiento con early stopping
- GIVEN train y validation construidos
- WHEN se entrena el modelo
- THEN se usa early_stopping_rounds=50 con la val como eval_set
- AND el modelo detiene el entrenamiento automáticamente para evitar overfit

### Requirement: Sample weights por temporada (recencia)

El sistema MUST soportar entrenamiento con `sample_weight` por fila, permitiendo dar más peso a las temporadas recientes. Se evalúan dos configuraciones: uniforme (peso=1 para todo) y decay por temporada.

#### Scenario: Experimento con pesos por temporada
- GIVEN train con 3 temporadas (2021/22, 2022/23, 2023/24)
- WHEN se entrena con pesos `{2021/22: 0.6, 2022/23: 0.8, 2023/24: 1.0}`
- THEN el modelo resultante se evalúa en validación y se compara con el entrenamiento de pesos uniformes
- AND en `metrics_v1.json` se guardan ambas configuraciones con sus MAE/RMSE

#### Scenario: Selección de la configuración final
- GIVEN las métricas de ambas configuraciones
- WHEN se selecciona el modelo a serializar
- THEN se elige la configuración con mejor MAE en validación
- AND se documenta la elección en el bloque de métricas

### Requirement: Sanity check con Negative Binomial

El sistema SHOULD entrenar un modelo NegBinom (statsmodels) sobre un subconjunto reducido de features como referencia estadística.

#### Scenario: Comparación NegBinom vs LightGBM
- WHEN ambos modelos están entrenados
- THEN se reportan MAE y RMSE de los dos en validación
- AND se documenta cuál es mejor y por qué

### Requirement: Métricas obligatorias

El sistema MUST reportar en validación y test: MAE, RMSE, MAE por equipo, MAE por temporada.

#### Scenario: Informe de evaluación
- WHEN el entrenamiento termina
- THEN imprime por consola un bloque con las 4 métricas
- AND guarda el informe en `data/model/metrics_v1.json`

### Requirement: Batir el baseline

El modelo final MUST superar el baseline de media histórica por equipo (MAE 4.84) en validación.

#### Scenario: Modelo no bate baseline
- GIVEN MAE del modelo en val >= 4.84
- WHEN termina el entrenamiento
- THEN el script emite warning claro indicando fallo de criterio de éxito
- AND no bloquea guardado (permite iteración) pero lo deja documentado en las metrics

#### Scenario: Modelo bate baseline
- GIVEN MAE < 4.84 en validación
- WHEN termina el entrenamiento
- THEN guarda modelo en `data/model/model_v1.joblib` y metrics en `data/model/metrics_v1.json`

### Requirement: Feature importance

El sistema MUST exportar el ranking de feature_importance del modelo LightGBM.

#### Scenario: Importance guardado
- WHEN termina el entrenamiento
- THEN `data/model/feature_importance.csv` contiene las features ordenadas por gain

### Requirement: Selección de features por SHAP

El sistema MUST soportar entrenamiento con un subconjunto curado de features seleccionadas
por análisis SHAP offline, controlado por el flag `--features {all, shap30}` (default `shap30`).

#### Scenario: Entrenamiento con shap30 (default)
- GIVEN `SHAP_SELECTED_FEATURES` declarado en `model/features.py` (35 features desde 2026-04-20, event-based-features)
- WHEN se ejecuta `python -m model.train` (sin flag `--features`)
- THEN el modelo entrena con las features declaradas en `SHAP_SELECTED_FEATURES`
- AND `metrics_v1.json` incluye `feature_selection_method: "shap30"` y el nº real de features usado
- AND `model_v1.joblib` guarda la lista de features en `artifact["features"]`

#### Scenario: Entrenamiento con todas las features
- GIVEN el flag `--features all`
- WHEN se ejecuta `python -m model.train --features all`
- THEN el modelo entrena con el conjunto completo filtrado por leakage (~198 features anti-leakage desde event-based-features)
- AND `feature_selection_method: "all"` queda registrado en metrics

#### Scenario: Resultado del experimento SHAP shap30 (2026-04-20, pre-event-based)
- GIVEN dataset con temporadas 2021-2025 (1139 filas de train)
- WHEN se entrenó con shap30 y weights=both
- THEN val MAE (decay) = 4.3255 vs baseline 4.3379 → MEJORA −0.0124 ✅
- AND train_val_gap = 0.74 (vs 0.68 con 101 features) → menos overfitting relativo
- AND best_iteration = 295 (vs 255) → más profundidad, menos ruido
- DECISION: shap30 es el default de producción; --features all como fallback

#### Scenario: Resultado del experimento event-based-features (2026-04-20)
- GIVEN dataset con event-based-features integradas (255 columnas, 198 anti-leakage disponibles)
- WHEN se reentrenó con SHAP top-35 (15 event-based + 20 anteriores supervivientes)
- THEN val MAE = 3.9257 vs baseline anterior 4.3255 → MEJORA −0.3998 (−9.2%) ✅
- AND `std_y` (dispersión lateral del equipo) es el feature con mayor importancia SHAP (4× sobre el segundo)
- AND 15 de las 35 SHAP_SELECTED_FEATURES son derivadas de event-based-features (crosses, long_balls, heads, wide_ratio, avg_pass_length, avg_zone_x, std_y y sus rollings/ewma/opp)
- DECISION: ADOPTADO. SHAP_SELECTED_FEATURES actualizado a 35 features.

### Requirement: Walk-forward cross-validation diagnóstico

El sistema MAY ejecutar un walk-forward CV de 3 folds con `python -m model.train --mode cv` para
estimar la varianza del `val_MAE` entre temporadas. Este modo NO sobrescribe `model_v1.joblib`
ni `metrics_v1.json` — es puramente diagnóstico.

Los folds son de expanding window:
- Fold 1: train = {21/22, 22/23} → val = 23/24
- Fold 2: train = {21/22, 22/23, 23/24} → val = 24/25 (= split canónico)
- Fold 3: train = {21/22…24/25} → val = 25/26 (parcial, excluido de agregados)

#### Scenario: Ejecución básica del CV
- WHEN se ejecuta `python -m model.train --mode cv`
- THEN se entrenan 3 modelos LightGBM Poisson con la misma config que el modelo principal
- AND se genera `data/model/cv_results.json` con métricas por fold
- AND se genera `data/model/cv_results.csv` con una fila por fold + agregados
- AND NO se modifican `model_v1.joblib` ni `metrics_v1.json`

#### Scenario: Métricas reportadas por fold
- WHEN un fold completa el entrenamiento
- THEN el artefacto incluye `val_mae`, `val_rmse`, `train_mae`, `train_val_gap`,
  `best_iteration`, `baseline_mae`, `n_train`, `n_val` para ese fold

#### Scenario: Agregados solo sobre folds completos
- GIVEN Fold 3 marcado como `is_partial=true` (2025/26 en curso)
- WHEN se calculan agregados
- THEN `mean_val_mae`, `std_val_mae`, `min_val_mae`, `max_val_mae`, `range_val_mae`
  se computan sobre Folds 1-2 únicamente
- AND Fold 3 se reporta por separado

#### Scenario: Interpretación de la varianza
- WHEN el CV termina
- THEN el log imprime "std_val_MAE = X → cambios con delta < 2·X son ruido estadístico"
- AND el artefacto incluye el campo `significant_delta_2sigma`

### Requirement: Diagnóstico de overfitting (train_MAE y gap)

El sistema MUST calcular y registrar `train_MAE` y `train_val_gap = train_MAE / val_MAE` para cada configuración entrenada.

#### Scenario: Gap saludable
- WHEN el entrenamiento termina
- THEN `metrics_v1.json` incluye `train_mae` y `train_val_gap` para cada configuración LightGBM
- AND el log imprime explícitamente ambos valores para diagnóstico visual

### Requirement: Modelos cuantil (Q25/Q50/Q75)

El sistema MAY entrenar tres modelos LightGBM con `objective="quantile"` (α=0.25, 0.50, 0.75)
ejecutando `python -m model.train_quantile`. Usan las mismas 30 SHAP features y la misma
regularización que el modelo principal. Son complementarios (no sustituyen) a `model_v1.joblib`.

#### Scenario: Entrenamiento cuantil
- GIVEN `data/model/dataset.parquet` y `SHAP_SELECTED_FEATURES`
- WHEN se ejecuta `python -m model.train_quantile`
- THEN se generan `model_q25.joblib`, `model_q50.joblib`, `model_q75.joblib` en `data/model/`
- AND se genera `data/model/metrics_quantile.json` con pinball loss por cuantil,
  cobertura empírica Q25-Q75 y accuracy en líneas O/U típicas
- AND `model_v1.joblib` NO se modifica

#### Scenario: Calidad de los modelos cuantil
- WHEN el entrenamiento cuantil termina
- THEN el pinball loss de cada modelo < pinball loss del baseline trivial (predecir siempre la media)
- AND la cobertura empírica Q25-Q75 a nivel partido ≥ 40% (objetivo: 40-60%)
- AND el log imprime resumen de señales OVER/UNDER por línea de apuesta

#### Scenario: Resultado empírico (2026-04-20, val 2024/25)
- GIVEN dataset con temporadas 2021-2025, 30 SHAP features
- WHEN se entrenó con decay weights y early stopping
- THEN pinball Q25=1.67 | Q50=2.19 | Q75=1.82 vs baseline=2.37 → todos baten baseline ✅
- AND cobertura Q25-Q75 (partido) = 46.8% ✅ (teórico 50%)
- AND señal UNDER Q75<45.5: 186 casos, 76.3% hit rate | Q75<43.5: 117 casos, 70.1%

### Requirement: Modo bivariate (experimental)

El sistema MAY entrenar un pipeline bivariate paralelo que predice `total_throw_ins` a nivel de partido (Model1, LightGBM Poisson) más un factor de reparto lineal (`share_home`).

#### Scenario: Entrenamiento bivariate activado
- GIVEN el flag `--target bivariate` en la CLI
- WHEN se ejecuta `python -m model.train --target bivariate`
- THEN se carga `data/model/dataset_match.parquet` (1 fila por partido)
- AND se entrena Model1 con mismos hiperparámetros regularizados que el modo per-team
- AND se ajusta `sklearn.LinearRegression` con features `possession_diff` y `home_rolling_diff`
- AND se guarda `data/model/model_v1_total.joblib` y `data/model/share_coefs.json`
- AND se calcula el per-team MAE reconstruido (`pred_total × pred_share`)

#### Scenario: Comparación con modelo productivo
- WHEN el modo bivariate termina
- THEN `data/model/metrics_bivariate.json` incluye `per_team_mae_reconstructed` y `beats_current` (bool vs 4.3379)
- AND el log imprime `[MEJORA / NO MEJORA]` explícitamente

#### Scenario: Resultado del experimento (2026-04-20)
- GIVEN el dataset con temporadas 2021-2025
- WHEN se entrenó en modo bivariate
- THEN total_MAE val = 6.7574 (vs ~6.71 suma per-team actual)
- AND per_team_MAE_reconstructed = 4.3970 (vs 4.3379 actual) → NO MEJORA
- AND share_pred_std = 0.0132 (vs share_real_std = 0.0936): la regresión lineal colapsa a constante
- DECISION: modelo productivo permanece como per-team (`model_v1.joblib`)
