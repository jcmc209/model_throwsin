# Exploration: weather-features

## Current State

El proyecto dispone de datos de partido (team_stats, throw_ins) para LaLiga 2021/22–2025/26. No existe ninguna fuente de datos meteorológicos. Los ficheros de referencia disponibles son:

- `data/reference/stadiums.csv` — 46 estadios con lat/lon, dimensiones de campo y `whoscored_id`
- `data/reference/liga_calendar_rows.csv` — 1.563 filas, temporadas 2023/24–2025/26, con `match_time` (hora de kickoff) y `status` (finished / scheduled / postponed)

**Datos históricos en team_stats**: 1.811 partidos únicos, rango 2021-08-13 → 2026-04-13.  
**Partidos futuros (scheduled, LaLiga)**: 100 en el calendario.

### Problema crítico: ausencia de hora en team_stats (2021/22 y 2022/23)

`team_stats.match_date` solo contiene fecha (`2021-08-14T00:00:00`), sin hora de kickoff. El calendario cubre desde 2023/24. Las temporadas 2021/22 y 2022/23 (~760 partidos) no tienen hora conocida.

### Problema de nombres: calendar → stadiums (16 mismatches)

| Calendar name | Stadiums club | Mapping |
|---|---|---|
| Atlético Madrid | Atlético de Madrid | manual |
| Barcelona | FC Barcelona | manual |
| Celta Vigo | RC Celta | manual |
| Cádiz | Cádiz CF | manual |
| Deportivo Alavés | Alavés | manual |
| Elche | Elche CF | manual |
| Espanyol | RCD Espanyol | manual |
| Getafe | Getafe CF | manual |
| Granada | Granada CF | manual |
| Las Palmas | UD Las Palmas | manual |
| Leganés | CD Leganés | manual |
| Mallorca | RCD Mallorca | manual |
| Osasuna | CA Osasuna | manual |
| Sevilla | Sevilla FC | manual |
| Valencia | Valencia CF | manual |
| Villarreal | Villarreal CF | manual |

10 equipos coinciden exactamente (Athletic Club, Girona FC, Levante UD, Rayo Vallecano, Real Betis, Real Madrid, Real Oviedo, Real Sociedad, Real Valladolid, Almería).

## Affected Areas

- `data/reference/stadiums.csv` — añadir columna `calendar_name` para los 16 mismatches
- `data/reference/weather.parquet` — nuevo fichero a crear (salida del fetcher)
- `weather_fetcher.py` — nuevo script (paralelo a `whoscored_scraper.py`)
- `openspec/config.yaml` — actualizar context con nueva fuente de datos

## API: Open-Meteo (confirmado funcional)

- **Histórico**: `https://archive-api.open-meteo.com/v1/archive` — sin API key, cubre desde 1940
- **Forecast**: `https://api.open-meteo.com/v1/forecast` — sin API key, 16 días adelante
- Respuesta: JSON con arrays horarios (`time`, `temperature_2m`, `wind_speed_10m`, etc.)
- Rate limit: ~10.000 req/día en tier gratuito — no es un problema

### Variables a extraer (por hora de kickoff)

| Variable | Unidad | Relevancia |
|---|---|---|
| `temperature_2m` | °C | Condición física general |
| `wind_speed_10m` | km/h | Efecto directo en saques de banda |
| `precipitation` | mm | Campo resbaladizo, más salidas por línea |
| `relative_humidity_2m` | % | Complementa temperatura |
| `weather_code` | WMO code | Categoría codificada (lluvia, nublado, etc.) |

## Approaches

### Approach 1 — Batching por estadio + temporada (recomendado)

Una sola llamada API por estadio por temporada (date_start → date_end), recuperando todos los días a la vez. Luego se filtra por fecha y hora de kickoff en pandas.

- **Pros**: ~100 llamadas totales (20 estadios × 5 temporadas) vs 1.811 individuales; muy rápido; fácil de relanzar por temporada
- **Cons**: Descarga datos de días sin partido (irrelevante para almacenamiento)
- **Esfuerzo**: Medio

### Approach 2 — Llamada individual por partido

Una llamada API por `match_id` con fecha exacta.

- **Pros**: Más simple de implementar; datos mínimos
- **Cons**: 1.811+ llamadas, lento, más puntos de fallo; difícil de resumir
- **Esfuerzo**: Bajo (código), Alto (tiempo de ejecución)

## Decisiones de diseño a tomar

### D1: ¿Qué hacer con 2021/22 y 2022/23 (sin hora de kickoff)?

| Opción | Descripción | Impacto en modelo |
|---|---|---|
| **A** — Hora fija | Asumir 20:00 para todos | Introduce ruido pequeño (±2h), simple |
| **B** — Agregado diario | Usar media/máx del día entero | Pierde contexto horario, más robusto |
| **C** — Excluir del dataset meteorológico | Solo 2023/24+ tienen weather | Pierde 760 partidos (42% del dataset histórico) |

**Recomendación**: Opción A (hora fija 20:00). La mayoría de partidos en España se juegan entre 18:30 y 21:00. El error máximo es de ~2h. Para temperatura y precipitación, el impacto es marginal.

### D2: ¿Clave de join?

`team_stats` tiene `match_id` (WhoScored). El calendario tiene `event_id` (externo). No comparten clave directamente. El join se hace por `(match_date_only, home_team_id)`.

Pasos:
1. Calendar: `home_team` → `calendar_name` en stadiums → `whoscored_id` (= `team_id`)
2. Join: `(match_date[:10], team_id_home)` entre team_stats y calendar → obtenemos `match_time`
3. Con `(lat, lon, date, hour)` → llamada a Open-Meteo

### D3: Estadios con cambio de nombre (2021→2026)

- Metropolitano: era "Wanda", ahora "Riyadh Air" — mismo campo, mismas coordenadas. Sin impacto.
- Camp Nou: en obras (2023/24), partidos en Olímpico de Montjuïc (lat 41.3641, lon 2.1538). **Necesita manejo especial** para 2023/24.
- Son Moix (Mallorca): renombrado pero mismo campo. Sin impacto.

## Recommendation

**Approach 1** (batching por estadio + temporada) con **D1-A** (hora fija 20:00 para temporadas sin calendar) y **D2** como join strategy.

Script `weather_fetcher.py` con:
- Resume incremental: skip estadios+temporadas ya descargados
- Manejo especial de Barcelona 2023/24 (Olímpico de Montjuïc)
- Columna `calendar_name` añadida a `stadiums.csv`
- Output: `data/reference/weather.parquet` con `match_id` como clave

## Risks

- **Camp Nou 2023/24**: Si no se maneja, los datos meteorológicos de Barcelona ese año serán del Camp Nou (distancia ~4km al Olímpico — error pequeño pero corregible)
- **Partidos aplazados**: 9 `postponed` en el calendar — hay que excluirlos o tratarlos como la nueva fecha
- **16-day forecast limit**: Open-Meteo forecast cubre solo 16 días. Partidos más lejanos (>16 días) no tendrán forecast hasta que estén dentro del rango
- **Cambio de estadio infrecuente**: Si un equipo juega en otro campo (sanciones, obras), las coordenadas del estadio habitual serán incorrectas — riesgo bajo y no corregible sin fuente adicional

## Ready for Proposal

Sí. Hay una sola decisión abierta pendiente de confirmar con el usuario: **el manejo de Camp Nou 2023/24**. Todo lo demás está suficientemente claro para escribir el proposal.
