# Exploration: recent-ctx-perf

## Current State

Hay dos problemas de rendimiento independientes en `whoscored_scraper.py`:

---

### Problema A — `_recent_ctx` es O(n²) por partido

`_recent_ctx(events, current_idx, team_id, window=5)` (líneas 537-573) se llama
**una vez por cada throw-in** en `_build_throw_ins` (línea 656).

Para cada llamada escanea **todos los eventos anteriores** dos veces:
```python
team_ev = [e for e in events[:current_idx]   # hasta current_idx eventos
           if e.get("teamId") == team_id
           and e.get("expandedMinute", ...) >= lo]
all_ev  = [e for e in events[:current_idx]   # ídem
           if e.get("expandedMinute", ...) >= lo]
```

Complejidad efectiva: **O(throw_ins × events_per_match)**

Números reales:
- Partido típico: ~2000 eventos, ~50 throw-ins
- Iteraciones por partido: ~50 × 2000 = **100.000 comparaciones Python**
- A escala de 1520 partidos (4 temporadas): **~150M iteraciones**

La ventana de 5 minutos restringe los eventos relevantes a ≈100-200, pero el
scan sigue siendo sobre todos los eventos anteriores (hasta 2000) para luego
filtrar los que caen en la ventana.

---

### Problema B — `plot_pass_map` usa `iterrows()`

`plot_pass_map` (línea 1371) itera fila a fila con `df.iterrows()` para dibujar
flechas. Un partido completo tiene ~800-1000 pases. `iterrows()` en pandas es
~100x más lento que operaciones vectorizadas.

`pitch.arrows` de `mplsoccer` acepta **arrays completos** como argumentos
(`x`, `y`, `end_x`, `end_y`), por lo que el loop es completamente innecesario.

---

## Affected Areas

- `whoscored_scraper.py` L537-573 — `_recent_ctx`: scan O(n) por llamada
- `whoscored_scraper.py` L656 — `_build_throw_ins`: llama `_recent_ctx` por cada throw-in
- `whoscored_scraper.py` L1371-1377 — `plot_pass_map`: `iterrows()` innecesario

## Approaches

### Problema A — tres opciones

**A1. Pre-filtrar por ventana de tiempo antes del loop de throw-ins**
Antes del loop `for idx, ev in enumerate(events)`, construir una lista de
eventos indexados por minuto y usar búsqueda binaria para obtener la ventana.
- Pros: O(n log n) total, sin cambiar la interfaz de `_recent_ctx`
- Cons: requiere estructura auxiliar
- Esfuerzo: Medio

**A2. Precalcular contexto acumulativo con sliding window**
Recorrer `events` una sola vez manteniendo contadores deslizantes por equipo.
Para cada evento se actualizan los contadores; al encontrar un throw-in se
captura el estado actual. Complejidad: **O(n)** total por partido.
- Pros: óptimo, elimina completamente `_recent_ctx`
- Cons: refactor más profundo, lógica de ventana deslizante no trivial
- Esfuerzo: Medio-Alto

**A3. Acotar el scan en `_recent_ctx` con búsqueda desde el final**
En lugar de `events[:current_idx]`, iterar hacia atrás desde `current_idx`
y parar en cuanto el minuto sea `< lo`. La mayoría de partidos tienen
pocos eventos de throw-in en la mitad del partido, así el scan real es
de ~100-200 eventos en vez de ~2000.
- Pros: cambio mínimo, mismo resultado, reducción real de ~10-20x
- Cons: no elimina la O(n²) en el peor caso teórico
- Esfuerzo: Bajo

### Problema B — vectorización

**B1. Reemplazar `iterrows()` por llamada vectorizada a `pitch.arrows`**
`mplsoccer` acepta arrays; llamar una vez con columnas completas del DataFrame.
Para colores y anchos (que dependen de cada fila), segmentar el DataFrame en
grupos (exitosos/fallidos, throw-ins/normales) y hacer una llamada por grupo.
- Pros: eliminación total del loop, ~100x más rápido en datasets grandes
- Cons: la lógica de color/alpha/lw requiere segmentación del DataFrame
- Esfuerzo: Bajo

## Recommendation

- **Problema A → Opción A3**: scan desde el final. Cambio mínimo, seguro,
  mejora real de ~10-20x sin reescribir la lógica de negocio. A2 sería
  óptimo pero el riesgo de introducir errores en los cálculos de contexto
  es mayor.
- **Problema B → Opción B1**: vectorización completa. Cambio pequeño y
  claramente correcto.

## Risks

- A3: si los eventos no están ordenados por minuto, el early-exit podría
  cortar antes de tiempo. Los datos de WhoScored llegan ordenados
  cronológicamente, pero conviene verificarlo o añadir una guarda.
- B1: `pitch.arrows` vectorizado dibuja en un orden distinto al loop;
  visualmente idéntico salvo en casos de solapamiento de flechas (irrelevante).

## Ready for Proposal

**Sí.** Dos mejoras independientes, bien acotadas, sin cambio de contrato público.
