# prediction Specification

## Purpose

Define cómo se generan predicciones para partidos futuros usando el modelo entrenado. Cubre construcción de features en tiempo de inferencia, manejo de casos límite (sin weather, sin historial) y formato de salida.

## Requirements

### Requirement: Input desde el calendario

El sistema MUST aceptar como input una lista de `event_id` del calendario con status `scheduled` o una fecha, y generar features consistentes con las de entrenamiento.

#### Scenario: Predicción de jornada futura
- GIVEN los partidos scheduled del calendario para la próxima jornada
- WHEN se ejecuta `predict.py`
- THEN se genera una fila por partido × equipo (home y away)
- AND cada fila tiene las mismas features que el dataset de entrenamiento

### Requirement: Features construidas con datos estrictamente anteriores

El sistema MUST construir las features rolling usando SOLO partidos con fecha < fecha del partido a predecir.

#### Scenario: Partido de la jornada 30, datos hasta jornada 29
- GIVEN un partido scheduled en fecha F
- WHEN se construyen sus features
- THEN los rolling usan partidos del equipo y del oponente con fecha < F
- AND ningún partido de fecha >= F se incluye

### Requirement: Manejo de partido sin forecast meteorológico

El sistema MUST manejar partidos a más de 16 días vista (sin forecast de Open-Meteo disponible).

#### Scenario: Predicción con weather disponible
- GIVEN un partido dentro de los próximos 16 días
- WHEN se construyen las features
- THEN las columnas meteorológicas se rellenan con datos reales de forecast

#### Scenario: Predicción con weather faltante
- GIVEN un partido a más de 16 días vista
- WHEN se construyen las features
- THEN las columnas meteorológicas se rellenan con la media histórica del estadio en el mes del partido
- AND la fila incluye `weather_imputed=True` para trazabilidad

### Requirement: Output estructurado

El sistema MUST devolver las predicciones como DataFrame con columnas: `event_id`, `match_date`, `home_team`, `away_team`, `pred_throw_ins_home`, `pred_throw_ins_away`, `pred_throw_ins_total`.

#### Scenario: Guardado en disco
- WHEN las predicciones están calculadas
- THEN se guardan en `data/model/predictions_<YYYYMMDD>.parquet`
- AND se imprime tabla resumida por consola

### Requirement: Modelo cargado desde disco

El sistema MUST cargar el modelo serializado en `data/model/model_v1.joblib` y usar su versión sin reentrenar.

#### Scenario: Modelo faltante
- GIVEN no existe `model_v1.joblib`
- WHEN se ejecuta `predict.py`
- THEN aborta con mensaje claro indicando ejecutar `train.py` primero

### Requirement: Modo cuantil de inferencia (Q25/Q50/Q75)

El sistema MAY generar predicciones con intervalos de confianza usando `--mode quantile`.
Carga los tres modelos cuantil y produce columnas `pred_q25/q50/q75` por equipo y totales
`total_q25/q50/q75` a nivel partido con clip anti-cruce aplicado.

#### Scenario: Inferencia cuantil activada
- GIVEN los modelos `model_q{25,50,75}.joblib` existen en `data/model/`
- WHEN se ejecuta `python -m model.predict --mode quantile --matchday next`
- THEN se aplica el mismo pipeline de features que el modo estándar
- AND se generan predicciones Q25/Q50/Q75 por equipo; se agregan a nivel partido
- AND se aplica clip: `total_Q25 ≤ total_Q50 ≤ total_Q75` para evitar cruces
- AND se guardan en `data/model/predictions_quantile_<YYYYMMDD>.parquet`
- AND se imprime tabla con `total_q25`, `total_q50`, `total_q75` por partido

#### Scenario: Estrategia de apuesta derivada
- GIVEN predicciones cuantiles para un partido y una línea de apuesta L
- WHEN se compara `total_Q25` y `total_Q75` con L
- THEN si `total_Q25 > L` → señal OVER (75% confianza histórica)
- AND si `total_Q75 < L` → señal UNDER (75% confianza histórica)
- AND si `total_Q25 ≤ L ≤ total_Q75` → incertidumbre alta, no apostar

#### Scenario: Modelos cuantil faltantes
- GIVEN no existen los archivos `model_q*.joblib`
- WHEN se ejecuta con `--mode quantile`
- THEN aborta con mensaje claro indicando ejecutar `python -m model.train_quantile` primero

### Requirement: Modo bivariate de inferencia (experimental)

El sistema MAY generar predicciones usando el pipeline bivariate (`--bivariate`) cargando `model_v1_total.joblib` + `share_coefs.json`.

#### Scenario: Inferencia bivariate activada
- GIVEN el flag `--bivariate` en la CLI
- WHEN se ejecuta `python -m model.predict --bivariate --matchday next`
- THEN se construyen las features en formato match-level (pivote home/away en memoria)
- AND se predice `pred_total` con Model1 y `pred_share` con el share factor lineal
- AND `pred_home = pred_total × pred_share`, `pred_away = pred_total × (1 - pred_share)`
- AND el output tiene el mismo schema que el modo per-team: `match_id`, `match_date`, `home_team`, `away_team`, `pred_throw_ins_home`, `pred_throw_ins_away`, `pred_throw_ins_total`
- AND se guarda en `data/model/predictions_bivariate_<YYYYMMDD>.parquet`
