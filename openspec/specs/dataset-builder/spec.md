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
