# Exploration: model-pipeline

## Current State

El proyecto tiene datos de 5 temporadas de LaLiga (2021/22–2025/26) completamente scrapeados y procesados.
No existe ningún script de modelado — el pipeline empieza desde cero.

### Datos disponibles y sus roles

| Fichero | Filas | Rol en pipeline |
|---|---|---|
| `team_stats.parquet` (×5 temporadas) | 3.622 (1.811 partidos × 2 equipos) | Base: target + features de historial |
| `weather.parquet` | 1.811 | Features pre-partido: temperatura, viento, lluvia |
| `stadiums.csv` | 26 equipos | Features estáticas: dimensiones campo, aforo |
| `throw_ins.parquet` (×5 temporadas) | ~65k eventos | Features avanzadas (zona, lado, minuto) — fase 2 |

### Target

| Objetivo | Descripción | Distribución |
|---|---|---|
| `y_home` | throw_ins_total del local | Media 20.6, std 6.1, rango [4, 47] |
| `y_away` | throw_ins_total del visitante | Media 19.4, std 5.8, rango [4, 47] |
| `y_total` | suma (mercado over/under) | Media 40.0, std 9.4, rango [15, 71] |

Distribución compatible con **Negative Binomial** (conteo discreto con sobredispersión: std > sqrt(media)).

### Problema crítico: data leakage

Las 34 columnas numéricas de `team_stats` (shots, passes, corners, etc.) son **estadísticas del partido actual** — no se conocen antes del partido. Usarlas como features directas provocaría leakage total.

**Solución**: construir features rolling (ventana deslizante sobre partidos anteriores del equipo) para cada una de esas columnas. La correlación del rolling5 de `throw_ins_total` con el target ya es 0.239, mejor que cualquier feature estática.

### Correlaciones observadas con `throw_ins_total` (home)

| Feature | Correlación | Tipo |
|---|---|---|
| `rolling5_throw_ins_home` | **+0.239** | Rolling histórico |
| `possession_pct` (partido) | +0.210 | ⚠️ Leakage |
| `pitch_length_m` | -0.159 | Estática |
| `attendance` | -0.140 | Pre-partido |
| `capacity` | -0.125 | Estática |
| `temperature_2m` | -0.084 | Pre-partido |
| `wind_speed_10m` | +0.069 | Pre-partido |

Las correlaciones individuales son bajas — el modelo necesita **interacciones entre features** (estilo equipo local vs visitante, campo pequeño con equipos defensivos, etc.).

### Baseline calculado (validación 2024/25)

| Baseline | MAE | RMSE |
|---|---|---|
| Media global de la temporada | 5.156 | 6.354 |
| Media histórica por equipo | **4.843** | **5.990** |

Cualquier modelo debe superar MAE < 4.84 en validación para ser útil. Un modelo rolling bien construido debería llegar a MAE ~4.0–4.5.

### Split temporal

```
Train:      2021/22 + 2022/23 + 2023/24  (1.139 partidos únicos)
Validation: 2024/25                       (380 partidos)
Test:       2025/26 (jugados)             (292 partidos)
```

Walk-forward: no shuffle. Los partidos de 2025/26 aún en curso sirven de test real.

## Affected Areas

- `model/` — directorio nuevo (no existe)
- `model/dataset_builder.py` — carga, join, feature engineering, exporta X/y
- `model/train.py` — entrenamiento, evaluación, serialización del modelo
- `model/predict.py` — inferencia para partidos futuros
- `data/model/` — directorio nuevo para outputs del pipeline
- `openspec/config.yaml` — actualizar con nueva referencia a `model/`

## Approaches

### Approach 1 — Scripts modulares en `model/` (recomendado)

```
model/
├── dataset_builder.py  ← ensambla X/y con todas las features
├── train.py            ← entrena y evalúa
└── predict.py          ← predice partidos futuros
```
- **Pros**: reproducible, versionable, fácil de relanzar; cada script independiente; encaja con el estilo del proyecto (whoscored_scraper.py, weather_fetcher.py)
- **Cons**: menos interactivo que un notebook para exploración inicial
- **Esfuerzo**: Medio

### Approach 2 — Jupyter Notebook exploratorio primero

Notebook `01_eda_model.ipynb` para explorar features, luego refactorizar a scripts.
- **Pros**: mejor para EDA, visualizaciones rápidas, iteración rápida
- **Cons**: notebook no es reproducible como pipeline; hay que refactorizar después
- **Esfuerzo**: Bajo (corto plazo), Medio (si hay que refactorizar)

### Approach 3 — Híbrido (recomendado real)

Notebook para EDA + scripts para pipeline productivo. El notebook vive en `notebooks/` y los scripts en `model/`.
- **Pros**: exploración libre + pipeline reproducible
- **Cons**: duplicación mínima entre notebook y scripts
- **Esfuerzo**: Medio

## Decisiones de diseño a resolver

### D1: ¿Un modelo o dos? (home vs away)

| Opción | Ventaja | Desventaja |
|---|---|---|
| **Dos modelos separados** (home_model, away_model) | Captura asimetría local/visitante | Doble mantenimiento |
| **Un modelo con `is_home` como feature** | Más simple, más datos de entrenamiento | Puede no capturar bien la asimetría |
| **Modelo total + Beta para reparto** | Encaja con mercado over/under | Más complejo |

Recomendación: **dos modelos separados** — la diferencia media (20.6 vs 19.4) y la naturaleza del mercado (apuestas por local y visitante por separado) lo justifican.

### D2: Features de historial rolling — ¿qué ventana?

- Rolling 3 partidos: más reactivo, mayor varianza
- Rolling 5 partidos: buen balance (confirmado correlación 0.239)
- Rolling 8 partidos (Sonnet sugería): más estable
- Recomendación: **rolling 5 como principal + rolling 10 como secundario**

### D3: Features de interacción de estilos

El feature más poderoso es probablemente **estilo local vs estilo visitante**: un equipo defensivo que fuerza saques vs uno que ataca por bandas. Esto requiere features rolling del oponente también (no solo del equipo).

### D4: Algoritmo base

| Algoritmo | Fit para conteo | Implementación |
|---|---|---|
| **Negative Binomial** (statsmodels) | Perfecto teóricamente | Media complejidad |
| **LightGBM Poisson** | Muy fuerte en práctica | Sencillo |
| **LightGBM Tweedie** | Bueno para distribuciones mixtas | Sencillo |

Recomendación: empezar con **LightGBM Poisson** (más fácil de tunear, probablemente mejor MAE) y comparar contra NegBinom como referencia estadística.

## Recommendation

**Approach 3 (híbrido)**: notebook `notebooks/01_eda.ipynb` para exploración + `model/dataset_builder.py` + `model/train.py` + `model/predict.py`.

Features del dataset:
1. Rolling stats del equipo (últimos 5 y 10 partidos): `throw_ins`, `possession`, `corners`, `passes`, `touches`
2. Rolling stats del oponente (últimos 5 partidos): mismas columnas
3. `is_home`, `team_id`, `opponent_id` (codificados)
4. Dimensiones de campo + aforo (del estadio local)
5. Meteorología: temperatura, viento, lluvia
6. Jornada de la temporada (fatiga acumulada)

Dos modelos: `model_home` y `model_away`, ambos LightGBM Poisson.
Métricas: MAE, RMSE, MAE relativo vs baseline de media por equipo.

## Risks

- **Rolling en primeros partidos de temporada**: equipos con pocos partidos previos reciben media global como fallback — introduce sesgo en agosto/septiembre
- **Equipos ascendidos**: primera temporada en LaLiga sin historial — dependen del fallback global
- **No estacionariedad**: la media de saques varía por temporada (21.7 en 21/22 vs 18.9 en 25/26) — el modelo puede estar sesgado hacia temporadas recientes
- **Multicolinealidad**: las features rolling de distintas columnas están correlacionadas entre sí — LightGBM lo maneja bien pero NegBinom puede fallar

## Ready for Proposal

Sí. Las decisiones clave están claras: hybrid approach, dos modelos, LightGBM Poisson, rolling 5+10, split temporal por temporada. Listo para propuesta.
