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
- THEN guarda modelo en `data/model/model_v1.pkl` y metrics en `data/model/metrics_v1.json`

### Requirement: Feature importance

El sistema MUST exportar el ranking de feature_importance del modelo LightGBM.

#### Scenario: Importance guardado
- WHEN termina el entrenamiento
- THEN `data/model/feature_importance.csv` contiene las features ordenadas por gain
