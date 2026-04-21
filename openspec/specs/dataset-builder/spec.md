# dataset-builder Specification

## Purpose

Define el comportamiento del builder que genera el dataset de modelado a partir de `team_stats`, `stadiums.csv` y `weather.parquet`. Produce un dataset sin data leakage, con features rolling y estáticas, listo para entrenamiento.

## Requirements

### Requirement: No data leakage

El dataset MUST NOT contener ninguna estadística del partido actual como feature. Solo columnas pre-partido, estáticas o rolling sobre partidos previos.

#### Scenario: Columna del partido actual como feature
- GIVEN una fila del dataset para el partido M del equipo T
- WHEN se construyen las features
- THEN ninguna columna calculada sobre estadísticas de M (shots, passes, etc.) aparece como feature
- AND `throw_ins_total` del partido M aparece SOLO como target, nunca como input

### Requirement: Rolling features del equipo y del oponente

El sistema MUST calcular, por cada fila, rolling averages de los N partidos previos del equipo y del oponente, con ventanas 3, 5 y 10.

#### Scenario: Rolling completo disponible
- GIVEN un equipo con ≥10 partidos jugados antes de la fecha del partido
- WHEN se construye la fila
- THEN las features `rolling{3,5,10}_*` contienen valores numéricos válidos

#### Scenario: Equipo ascendido sin historial suficiente
- GIVEN un equipo con <5 partidos previos (ej: ascendido, inicio de temporada)
- WHEN se construye la fila
- THEN las features rolling de ventanas mayores al historial disponible se rellenan con la media global de la temporada de training
- AND la fila incluye una columna `has_full_history` (bool) para que el modelo pueda ignorarla si quiere

### Requirement: EWMA features con decay exponencial

El sistema MUST calcular, para las mismas variables que rolling, medias móviles exponenciales (EWMA) que ponderen más los partidos recientes. Se generan dos decays distintos (α=0.3 y α=0.5) para que el modelo elija el que mejor explique.

#### Scenario: EWMA por equipo y oponente
- GIVEN un equipo con ≥3 partidos previos
- WHEN se construye la fila
- THEN existen las features `ewma_alpha03_*` y `ewma_alpha05_*` del propio equipo y del oponente
- AND el partido más reciente pesa más que los anteriores (decay exponencial)

#### Scenario: EWMA con pocos partidos
- GIVEN un equipo con <3 partidos previos
- WHEN se construye la fila
- THEN la EWMA se calcula con lo disponible (sin min_periods forzado)
- AND si no hay ningún partido previo, se imputa con la media global de training

### Requirement: Una fila por (partido, equipo)

El dataset MUST contener 2 filas por match_id (home y away), con la columna `is_home` ∈ {0, 1}.

#### Scenario: Cobertura completa
- GIVEN los 1.811 partidos de `team_stats`
- WHEN se ejecuta el builder
- THEN el dataset contiene 3.622 filas (1.811 × 2)
- AND cada `match_id` aparece exactamente dos veces

### Requirement: Join con meteorología y estadio

El sistema MUST unir el dataset base con `weather.parquet` (por `match_id`) y `stadiums.csv` (por `team_id` local) sin filas huérfanas.

#### Scenario: Join sin pérdida
- GIVEN el dataset base con 3.622 filas
- WHEN se hace left-join con weather y stadiums
- THEN el dataset resultante mantiene 3.622 filas
- AND las columnas `temperature_2m`, `wind_speed_10m`, `precipitation`, `pitch_length_m`, `pitch_width_m`, `capacity` no tienen nulls

### Requirement: Output serializado

El sistema MUST guardar el dataset en `data/model/dataset.parquet` con esquema estable.

#### Scenario: Re-ejecución idempotente
- GIVEN un dataset ya generado
- WHEN se re-ejecuta `dataset_builder.py`
- THEN el fichero se sobrescribe con el mismo esquema y tamaño
- AND los checksums de los valores numéricos son idénticos (determinismo)

### Requirement: Style features (índice de juego directo)

El sistema MUST calcular `rolling5_direct_play` y `opp_rolling5_direct_play` como columnas
del dataset. Estas features representan la media móvil de 5 partidos del índice
`aerials_total / (passes_total + 1)` del equipo y del oponente respectivamente.

Se calculan con anti-leakage (`shift(1)`) e imputación por media global. Se incluyen en el
dataset aunque no formen parte de `SHAP_SELECTED_FEATURES` en la versión actual del modelo.

#### Scenario: Cálculo sin data leakage
- GIVEN un equipo con ≥1 partido previo
- WHEN se llama a `compute_style_features(df)`
- THEN `rolling5_direct_play` contiene la media de hasta 5 partidos previos del índice
- AND nunca contiene estadísticas del partido actual

#### Scenario: Cobertura completa sin nulls
- GIVEN el dataset completo (3622 filas)
- WHEN se ejecuta el builder
- THEN `rolling5_direct_play` y `opp_rolling5_direct_play` tienen 0 nulls
- AND se imputa con la media global para los primeros partidos de cada equipo

### Requirement: Event-based features (acciones atómicas de partido)

El sistema MUST agregar las acciones atómicas de `all_events.parquet` por `(match_id, team_id)` y derivar features rolling/EWMA/std a partir de ellas. Estas features tienen vínculo mecánico directo con los saques de banda.

Los agregados base se calculan con `scripts/event_aggregator.py` y se guardan en `data/reference/event_stats.parquet`.

Las 8 columnas base son: `crosses`, `long_balls`, `heads`, `wide_events`, `wide_ratio`, `avg_pass_length`, `avg_zone_x`, `std_y`.

Se definen en la constante `EVENT_FEATURE_SOURCE_COLS` de `model/features.py` y se aplican las mismas funciones `compute_rolling`, `compute_ewma`, `compute_season_to_date` que a `FEATURE_SOURCE_COLS`.

#### Scenario: Generación de event_stats.parquet

- GIVEN los parquets `all_events` de las 5 temporadas disponibles (1811 partidos)
- WHEN se ejecuta `python scripts/event_aggregator.py`
- THEN se genera `data/reference/event_stats.parquet` con exactamente 3622 filas (1811 × 2 equipos)
- AND las columnas `crosses`, `long_balls`, `heads`, `wide_events`, `wide_ratio`, `avg_pass_length`, `avg_zone_x`, `std_y` no tienen nulls
- AND no hay duplicados por `(match_id, team_id)`

#### Scenario: Integración en dataset_builder

- GIVEN `event_stats.parquet` existente y `team_stats` cargado
- WHEN se ejecuta `build_dataset()`
- THEN `event_stats` se mergea en el dataset ANTES de `compute_rolling/ewma/std`
- AND se generan features `rolling{3,5,10}_*`, `ewma_alpha{03,05}_*`, `std_*` y sus contrapartes `opp_*` para cada columna de `EVENT_FEATURE_SOURCE_COLS`
- AND NaN en primeros partidos de cada equipo son aceptados (LightGBM los maneja nativamente)

#### Scenario: Impacto en rendimiento del modelo

- GIVEN el modelo reentrenado con las nuevas features (SHAP top-35 con 15 event-based)
- WHEN se evalúa en el conjunto de validación (temporada 2024/25)
- THEN el val MAE es 3.9257, una mejora de −0.3998 (−9.2%) respecto al baseline anterior de 4.3255
- AND `std_y` (dispersión lateral) emerge como la feature más importante con importancia SHAP 4× superior a la segunda

### Requirement: Match-level dataset para pipeline bivariate

El sistema MAY generar `data/model/dataset_match.parquet` (1 fila por partido) a partir de `dataset.parquet` mediante pivote por `is_home`.

#### Scenario: Construcción del match dataset
- GIVEN `data/model/dataset.parquet` con 3622 filas (1811 partidos × 2 equipos)
- WHEN se ejecuta `python -m model.dataset_builder_match`
- THEN se genera `data/model/dataset_match.parquet` con exactamente 1811 filas
- AND cada `match_id` aparece una sola vez
- AND las features per-team aparecen duplicadas con prefijos `home_` y `away_`
- AND las features match-level (capacity, weather, referee, matchday_number) se toman de la fila home sin duplicar
- AND se añaden los targets: `throw_ins_total_match`, `home_throw_ins_total`, `away_throw_ins_total`, `share_home`
