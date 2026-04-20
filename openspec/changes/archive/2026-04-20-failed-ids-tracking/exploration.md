# Exploration: failed-ids-tracking

## Current State

El pipeline principal (`run_scraper`) gestiona el progreso con un único fichero
`processed_ids.json` + la colección `team_stats` de Mongo. El flujo ante fallo es:

```
fetch_match_data() → None  (todos los reintentos agotados)
  → log.warning("✗ Sin datos: {match_id}")     # línea 1184
  → processed_ids.add(match_id)                 # línea 1185  ← BUG
  → _save_progress(progress_file, processed_ids) # línea 1186
  → continue                                     # salta al siguiente partido
```

**El problema**: un partido que falla por timeout, bloqueo de WhoScored o error
de red queda marcado en `processed_ids.json` **exactamente igual** que un partido
correctamente procesado. En el siguiente `run_scraper(..., resume=True)` ese ID
no aparece en `pending` y nunca se reintenta.

Hay un segundo caso de fallo silencioso: si `process_match()` lanza una excepción
(línea 1227-1228), el `except Exception` solo loguea el error pero el partido
igualmente se añade a `processed_ids` (línea 1230) y se guarda el progreso
(línea 1231). Datos procesados parcialmente, marcados como completos.

### Estructuras actuales de progreso

| Fichero / colección | Qué guarda | Problema |
|---------------------|-----------|---------|
| `{out_dir}/processed_ids.json` | Set de IDs "vistos" (éxito + fallo) | No distingue fallo de éxito |
| `MongoDB team_stats` | IDs con datos reales | Solo refleja éxitos con Mongo activo |
| `data/.../raw/{id}.json` | JSON crudo del partido | Solo existe si el scrape tuvo éxito |

`_save_progress` (línea 1249-1251) sobreescribe el fichero completo en cada
llamada; no hay append ni distinción de categoría.

### `fetch_match_data` — política de reintentos

- Máximo `CONFIG["max_retries"]` = 3 intentos
- Entre reintentos: `random_delay(15, 25)` segundos
- Tras agotar reintentos: devuelve `None`

No hay backoff exponencial ni categorización del tipo de error (timeout vs.
bloqueo vs. dato ausente).

## Affected Areas

- `whoscored_scraper.py` `run_scraper()` líneas 1183-1187 y 1224-1231 — lógica de fallo
- `whoscored_scraper.py` `_save_progress()` línea 1249 — función de escritura de progreso
- `{out_dir}/processed_ids.json` — fichero de estado en disco (formato cambia)

## Approaches

### 1. `failed_ids.json` separado (mínimo invasivo)
Añadir un segundo fichero `failed_ids.json` junto a `processed_ids.json`.
Los fallos se escriben allí en lugar de en `processed_ids`. Al reanudar,
`pending` excluye solo `processed_ids`; `failed_ids` se ofrece para reintento
manual o automático.
- Pros: cambio quirúrgico, no altera `processed_ids.json` existente, retrocompatible
- Cons: dos ficheros que mantener; `failed_ids` puede crecer sin limpieza
- Esfuerzo: Bajo

### 2. `processed_ids.json` con estructura `{id: status}`
Cambiar el formato de lista plana a dict `{match_id: "ok" | "failed"}`.
Resume solo salta los `"ok"`; los `"failed"` se reintentan.
- Pros: un único fichero, más expresivo
- Cons: rompe retrocompatibilidad con ficheros existentes (necesita migración), más cambio
- Esfuerzo: Medio

### 3. Estado en Mongo exclusivamente
Añadir una colección `scrape_log` con `{match_id, status, attempts, last_error}`.
- Pros: centralizado, consultable con queries
- Cons: depende de Mongo disponible; si Mongo cae, se pierde el tracking
- Esfuerzo: Alto

## Recommendation

**Opción 1** (`failed_ids.json` separado): cambio mínimo, sin romper ficheros
existentes, fácil de inspeccionar manualmente. El usuario puede borrar
`failed_ids.json` para forzar reintento de todos los fallos, o leerlo para
saber qué partidos necesitan atención.

Adicionalmente, corregir el segundo caso: cuando `process_match()` falla con
excepción, el partido NO debe añadirse a `processed_ids` (quedará en `pending`
en la siguiente ejecución y se reintentará automáticamente).

## Risks

- Si el fallo es permanente (partido eliminado de WhoScored), `failed_ids.json`
  crecerá indefinidamente. Mitigación: el usuario puede inspeccionarlo y
  decidir si omitir esos IDs definitivamente.
- Ficheros `processed_ids.json` existentes en `data/` no se ven afectados
  (no hay migración necesaria con la opción 1).

## Ready for Proposal

**Sí.** El problema es concreto, la solución está acotada, sin dependencias externas.
