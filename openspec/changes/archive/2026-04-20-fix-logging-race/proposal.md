# Proposal: fix-logging-race

## Intent

Al ejecutar `run_all_seasons.py`, el fichero `scraper_all_seasons.log` nunca se
crea. La causa: `whoscored_scraper.py` llama `logging.basicConfig` al importarse,
ocupando los handlers del root logger antes de que `run_all_seasons.py` pueda
configurar los suyos. El segundo `basicConfig` es un no-op silencioso.

## Scope

### In Scope
- Extraer la configuración de logging de `whoscored_scraper.py` a una función
  `setup_logging(log_file: str)`
- Llamar `setup_logging("scraper.log")` desde el bloque `__main__` de `whoscored_scraper.py`
- Reemplazar el `basicConfig` de `run_all_seasons.py` por una llamada a
  `setup_logging("scraper_all_seasons.log")`

### Out of Scope
- Logging estructurado (JSON), rotación de logs, niveles por módulo
- Cambios en el formato del mensaje de log

## Approach

Patrón estándar Python: los módulos importables no deben configurar el root
logger. `setup_logging(log_file)` centraliza el `basicConfig` y lo hace
explícito en cada punto de entrada.

## Affected Areas

| Área | Impacto | Descripción |
|------|---------|-------------|
| `whoscored_scraper.py` L74-82 | Modificado | `basicConfig` → función `setup_logging()` |
| `whoscored_scraper.py` bloque `__main__` | Modificado | Añadir llamada `setup_logging("scraper.log")` |
| `run_all_seasons.py` L17-24 | Modificado | `basicConfig` → `setup_logging("scraper_all_seasons.log")` |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|-----------|
| Scripts que importen el módulo sin llamar `setup_logging` pierden FileHandler | Baja | `test_single.py` y `validate.py` no usan FileHandler — stdout es suficiente |

## Rollback Plan

Revertir las 3 modificaciones. El comportamiento previo se restaura exactamente
(incluyendo el bug original).

## Dependencies

Ninguna externa.

## Success Criteria

- [ ] Ejecutar `python run_all_seasons.py` crea `scraper_all_seasons.log` en disco
- [ ] Ejecutar `python whoscored_scraper.py` sigue creando `scraper.log` en disco
- [ ] `test_single.py` y `validate.py` siguen ejecutando sin errores de logging
