# Proposal: failed-ids-tracking

## Intent

Los partidos que fallan durante el scraping (timeout, bloqueo, error de red) quedan
marcados como procesados y nunca se reintentan. Esto provoca pérdida silenciosa de
datos en runs largos sin ningún aviso de qué partidos faltan.

## Scope

### In Scope
- Añadir `failed_ids.json` junto a `processed_ids.json` para registrar fallos
- Corregir Caso 1: `fetch_match_data()` → `None` escribe en `failed_ids`, NO en `processed_ids`
- Corregir Caso 2: excepción en `process_match()` → partido NO se añade a `processed_ids`
  (queda en `pending` y se reintenta en el siguiente run)
- Añadir función auxiliar `_save_failed()` paralela a `_save_progress()`
- Log informativo al arrancar: cuántos IDs fallidos hay pendientes de reintento

### Out of Scope
- Cambiar formato de `processed_ids.json` (retrocompatibilidad preservada)
- Reintento automático dentro del mismo run
- Backoff exponencial en `fetch_match_data`
- Colección Mongo de scrape log

## Approach

Mínimo invasivo: dos ficheros separados, cada uno con su responsabilidad.
`processed_ids` = éxitos confirmados. `failed_ids` = fallos a revisar/reintentar.

Al reanudar, `pending` sigue siendo `match_ids - processed_ids`. Los IDs en
`failed_ids` vuelven a estar en `pending` automáticamente, permitiendo reintento.
Para omitir un fallo permanente, el usuario simplemente lo mueve a `processed_ids`
o borra la entrada de `failed_ids.json`.

## Affected Areas

| Área | Impacto | Descripción |
|------|---------|-------------|
| `whoscored_scraper.py` `run_scraper()` L1183-1187 | Modificado | Fallo de fetch → `failed_ids`, no `processed_ids` |
| `whoscored_scraper.py` `run_scraper()` L1227-1231 | Modificado | Excepción en process → no añadir a `processed_ids` |
| `whoscored_scraper.py` `_save_progress()` L1249 | Sin cambios | Sigue igual |
| `whoscored_scraper.py` | Nuevo | Función `_save_failed(path, ids)` |
| `{out_dir}/failed_ids.json` | Nuevo | Fichero de fallos por temporada |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|-----------|
| `failed_ids.json` crece con partidos permanentemente inaccesibles | Media | Usuario puede moverlos a `processed_ids` manualmente |
| Caso 2 corregido puede causar bucles si el error es sistemático | Baja | El log de error sigue visible; el usuario puede interrumpir |

## Rollback Plan

Revertir los 3 bloques modificados en `run_scraper()` y borrar `_save_failed()`.
Los `failed_ids.json` en disco son inofensivos si existen tras el rollback.

## Dependencies

Ninguna externa.

## Success Criteria

- [x] Partido con `fetch_match_data()` → `None` aparece en `failed_ids.json`, no en `processed_ids.json`
- [x] Partido con excepción en `process_match()` no aparece en `processed_ids.json` tras el run
- [x] En el siguiente run con `resume=True`, ambos tipos de fallos aparecen en `pending`
- [x] `processed_ids.json` existentes en `data/` siguen siendo legibles sin migración
