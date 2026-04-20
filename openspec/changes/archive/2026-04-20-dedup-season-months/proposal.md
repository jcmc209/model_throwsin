# Proposal: dedup-season-months

## Intent

`_get_season_months()` mantiene una tabla `season_start_year` que duplica datos
ya presentes en `CONFIG["known_seasons"]`. Añadir una temporada nueva exige editar
dos sitios; olvidar el segundo causa un fallback silencioso a 2024 y genera meses
de scraping incorrectos sin ningún error visible.

## Scope

### In Scope
- Eliminar la tabla `season_start_year` de `_get_season_months()`
- Cambiar la firma a `_get_season_months(season_key: str)` y derivar el año
  con `int(season_key[:4])`
- Actualizar la única llamada interna desde `get_match_ids()`

### Out of Scope
- Cambiar la firma pública de `get_match_ids()` (sigue recibiendo `season_id`, `stage_id`)
- Añadir validación de formato de `season_key`
- Refactor de CONFIG

## Approach

`season_key` tiene el formato `"YYYY/YYYY"` (convención estable). El año de inicio
es siempre `int(season_key[:4])`. Pasar la clave directamente a `_get_season_months`
elimina la tabla duplicada sin ningún acoplamiento adicional.

## Affected Areas

| Área | Impacto | Descripción |
|------|---------|-------------|
| `whoscored_scraper.py` L409-433 | Modificado | Eliminar `season_start_year`, nueva firma `(season_key: str)` |
| `whoscored_scraper.py` L275 | Modificado | Pasar `season_key` en lugar de `season_id` |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|-----------|
| `season_key` con formato inesperado rompe `int(season_key[:4])` | Muy baja | El formato `"YYYY/YYYY"` es validado al inicio de `run_scraper` |

## Rollback Plan

Revertir los dos bloques modificados. Sin impacto en datos ni en ficheros en disco.

## Dependencies

Ninguna externa.

## Success Criteria

- [ ] `_get_season_months` no contiene tabla `season_start_year`
- [ ] `_get_season_months("2025/2026")` devuelve los mismos 11 meses que antes
- [ ] Añadir una temporada a `CONFIG["known_seasons"]` es suficiente; no hay segundo sitio que editar
