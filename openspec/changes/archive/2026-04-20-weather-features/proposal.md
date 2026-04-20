# Proposal: weather-features

## Intent

Incorporar condiciones meteorológicas (temperatura, viento, precipitación, humedad) como features para el modelo predictivo de saques de banda. Cada partido histórico (2021/22–2025/26) recibirá los datos meteorológicos del estadio local en la hora del kickoff. Los partidos futuros usarán el forecast de Open-Meteo para hacer predicciones con contexto climático real.

## Scope

### In Scope
- Añadir columna `calendar_name` a `data/reference/stadiums.csv` (16 mappings manuales)
- Script `weather_fetcher.py`: descarga datos de Open-Meteo (histórico + forecast) y los guarda en `data/reference/weather.parquet`
- Batching por estadio+temporada (~100 llamadas totales, no 1.811 individuales)
- Resume incremental: skip estadios+temporadas ya descargados
- Output keyed por `match_id` para join directo con `team_stats`
- Hora de kickoff: extraída del calendario (2023/24+) o fija a 20:00 (2021/22 y 2022/23)

### Out of Scope
- Integración en el pipeline de modelado (fase posterior)
- Datos de estadios cubiertos vs descubiertos (todos los estadios de LaLiga son al aire libre)
- Manejo especial de Camp Nou 2023/24 (error de ~4km, impacto meteorológico despreciable — decisión del usuario)
- Partidos en estadio diferente al habitual por sanción u obras (no hay fuente de datos disponible)

## Approach

**Batching por estadio + temporada** via Open-Meteo API (gratuita, sin API key).

1. Añadir `calendar_name` a `stadiums.csv` para resolver los 16 mismatches de nombres
2. Para cada estadio × temporada: una llamada a `archive-api` (histórico) o `forecast` (futuro) con el rango de fechas completo de la temporada
3. Filtrar las filas horarias por `(match_date, kickoff_hour)` usando el calendario (2023/24+) o 20:00 fijo (anteriores)
4. Join con `team_stats` por `(match_date_only, home_team_id)` para asignar `match_id`
5. Guardar como `data/reference/weather.parquet`

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `data/reference/stadiums.csv` | Modified | +columna `calendar_name` (16 mappings) |
| `data/reference/weather.parquet` | New | Output principal: 1 fila por match_id |
| `weather_fetcher.py` | New | Script de descarga y procesado |
| `openspec/config.yaml` | Modified | +línea de referencia a weather.parquet |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Partidos aplazados con fecha incorrecta en calendar | Low | Filtrar `status != 'postponed'` |
| Partidos futuros >16 días sin forecast disponible | Med | Dejar `null`; relanzar fetcher antes de predecir |
| Match_id sin correspondencia en calendar (2021/22–2022/23) | Low | Esos partidos usan 20:00 fijo; join por fecha sin necesidad del calendar |

## Rollback Plan

- `weather.parquet` es un fichero nuevo — eliminarlo es rollback completo
- `stadiums.csv` se puede revertir eliminando la columna `calendar_name`
- `weather_fetcher.py` es un script independiente, no modifica ningún fichero existente de datos

## Dependencies

- `requests` (stdlib en Python 3.x, ya disponible)
- Open-Meteo API (pública, gratuita, sin registro)
- `data/reference/stadiums.csv` con lat/lon completos (ya disponible)
- `data/reference/liga_calendar_rows.csv` (ya disponible)
- `data/whoscored_laliga/**/team_stats.parquet` (ya disponible)

## Success Criteria

- [ ] `weather.parquet` cubre el 100% de los `match_id` en `team_stats` (sin nulls en `match_id`)
- [ ] Las 5 variables meteorológicas están presentes para todos los partidos históricos
- [ ] Los partidos `scheduled` dentro de 16 días tienen forecast (no null)
- [ ] El fetcher es re-ejecutable sin duplicar datos (idempotente)
- [ ] Join `team_stats.merge(weather, on='match_id')` no produce filas huérfanas
