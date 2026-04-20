# Proposal: recent-ctx-perf

## Intent

Dos cuellos de botella de rendimiento en el procesamiento de partidos:

1. `_recent_ctx` escanea hasta ~2000 eventos por cada throw-in (~50/partido)
   → ~100K comparaciones Python/partido → ~150M a escala de 4 temporadas.
2. `plot_pass_map` itera fila a fila con `iterrows()` sobre ~1000 pases cuando
   `pitch.arrows` de mplsoccer acepta arrays completos directamente.

## Scope

### In Scope
- **A3**: en `_recent_ctx`, iterar hacia atrás desde `current_idx` y parar
  en cuanto el evento quede fuera de la ventana de 5 minutos
- **B1**: en `plot_pass_map`, reemplazar el loop `iterrows()` por llamadas
  vectorizadas a `pitch.arrows` segmentando el DataFrame por color/ancho

### Out of Scope
- Sliding window acumulativo O(n) (A2) — refactor de mayor riesgo, diferido
- Cambios en la interfaz pública de `_recent_ctx` o `_build_throw_ins`
- Profiling formal con benchmarks automatizados

## Approach

**A3**: invertir el scan — `reversed(events[:current_idx])` — y hacer `break`
en cuanto `event_minute < lo`. Los eventos de WhoScored llegan ordenados
cronológicamente, así que el early-exit es correcto y efectivo.

**B1**: dividir el DataFrame en 4 grupos (exitoso/fallido × throw-in/normal),
llamar `pitch.arrows` una vez por grupo con arrays de columnas.

## Affected Areas

| Área | Impacto | Descripción |
|------|---------|-------------|
| `whoscored_scraper.py` L537-573 `_recent_ctx` | Modificado | Scan inverso con early-exit |
| `whoscored_scraper.py` L1371-1377 `plot_pass_map` | Modificado | `iterrows()` → vectorizado |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|-----------|
| Eventos desordenados rompen el early-exit de A3 | Muy baja | WhoScored devuelve eventos ordenados; el resultado es igual sin early-exit si el orden falla |
| B1 cambia orden de dibujo de flechas | Muy baja | Visualmente irrelevante |

## Rollback Plan

Revertir los dos bloques modificados. Sin impacto en datos, MongoDB ni outputs CSV/Parquet.

## Dependencies

Ninguna externa.

## Success Criteria

- [ ] `_recent_ctx` produce resultados idénticos a la versión actual para cualquier partido
- [ ] El scan de `_recent_ctx` no recorre más eventos que los necesarios para cubrir la ventana
- [ ] `plot_pass_map` no usa `iterrows()` y produce el mismo gráfico visual
