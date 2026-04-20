# Exploration: dedup-season-months

## Current State

### El problema

El mismo dato — la relación entre `season_id` y el año de inicio de temporada —
está definido en **dos sitios distintos**:

**Sitio 1** — `CONFIG["known_seasons"]` (líneas 49-57):
```python
"known_seasons": {
    "2025/2026": {"season_id": 10803, "stage_id": 24622},
    "2024/2025": {"season_id": 10247, "stage_id": 23815},
    "2023/2024": {"season_id":  9715, "stage_id": 23277},
    "2022/2023": {"season_id":  9098, "stage_id": 22580},
    "2021/2022": {"season_id":  8558, "stage_id": 21963},
    "2020/2021": {"season_id":  8016, "stage_id": 21413},
    "2019/2020": {"season_id":  7466, "stage_id": 20486},
}
```

**Sitio 2** — `season_start_year` dentro de `_get_season_months()` (líneas 415-423):
```python
season_start_year = {
    10803: 2025,  # 2025/2026
    10247: 2024,  # 2024/2025
    9715:  2023,  # 2023/2024
    9098:  2022,  # 2022/2023
    8558:  2021,  # 2021/2022
    8016:  2020,  # 2020/2021
    7466:  2019,  # 2019/2020
}
```

El año de inicio ya está implícito en la clave de `known_seasons`
(p.ej. `"2025/2026"` → primer año = 2025). Añadir una nueva temporada
exige editar ambos sitios; olvidar el segundo produce un fallback silencioso
a 2024 (`season_start_year.get(season_id, 2024)`), generando meses incorrectos
sin ningún error visible.

### Flujo de llamada

```
run_scraper(season_key)
  → CONFIG["known_seasons"][season_key]["season_id"]  → season_id
  → get_match_ids(page, season_id, stage_id)
      → _get_season_months(season_id)          ← recibe solo el int, no la clave
          → season_start_year[season_id]        ← tabla duplicada
```

La causa raíz está en que `get_match_ids` recibe `season_id` (int) pero no
`season_key` (str), perdiendo el contexto necesario para derivar el año
directamente de `CONFIG`.

## Affected Areas

- `whoscored_scraper.py` L409-433 — `_get_season_months`: tabla `season_start_year` a eliminar
- `whoscored_scraper.py` L265 — `get_match_ids`: firma podría recibir `season_key` en lugar de / además de `season_id`
- `whoscored_scraper.py` L1151-1152 — llamada a `get_match_ids` desde `run_scraper`

## Approaches

### 1. Derivar el año directamente de CONFIG en `_get_season_months`
Cambiar la firma: `_get_season_months(season_id)` → recorre `CONFIG["known_seasons"]`
para encontrar la clave cuyo `season_id` coincida, extrae el año del string de la clave.
- Pros: elimina la tabla duplicada completamente; una sola fuente de verdad
- Cons: acoplamiento de `_get_season_months` a CONFIG (hoy ya lo tiene indirectamente)
- Esfuerzo: Bajo

### 2. Pasar `season_key` a `_get_season_months` en lugar de `season_id`
Cambiar la firma: `_get_season_months(season_key: str)` y derivar el año del string
`"YYYY/YYYY"` con `int(season_key[:4])`.
- Pros: más explícito, sin búsqueda en CONFIG, sin tabla duplicada, firma más clara
- Cons: requiere ajustar la firma de `_get_season_months` y su llamada interna
- Esfuerzo: Bajo

### 3. Añadir `start_year` al diccionario de `known_seasons` en CONFIG
```python
"2025/2026": {"season_id": 10803, "stage_id": 24622, "start_year": 2025}
```
- Pros: CONFIG es la única fuente de verdad, explícito
- Cons: redundante (el año ya está en la clave); si alguien escribe mal el año,
  CONFIG tiene dos valores contradictorios
- Esfuerzo: Bajo

## Recommendation

**Opción 2**: pasar `season_key` a `_get_season_months` y extraer el año con
`int(season_key[:4])`. Es la solución más simple, más legible y sin dependencia
de búsqueda en CONFIG. El año de inicio siempre es el primer componente del string
de la clave (convención estable de WhoScored).

## Risks

- `_get_season_months` es función privada usada solo en `get_match_ids`; cambiar
  su firma no afecta a ningún caller externo.
- `get_match_ids` es función pública — su firma no cambia (sigue recibiendo
  `season_id` y `stage_id`); solo la llamada interna a `_get_season_months` cambia.

## Ready for Proposal

**Sí.** Refactor acotado a una función privada y su caller inmediato.
