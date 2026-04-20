# Exploration: fix-logging-race

## Current State

### El bug

`logging.basicConfig()` es una función **idempotente con trampa**: solo tiene efecto
la primera vez que se llama. Si el root logger ya tiene handlers configurados, las
llamadas posteriores son silenciosamente ignoradas.

Secuencia de ejecución cuando el usuario lanza `python run_all_seasons.py`:

```
1. Python importa whoscored_scraper (línea 15 de run_all_seasons.py)
   → load_dotenv() ejecuta (línea 38 de whoscored_scraper.py)
   → logging.basicConfig(handlers=[StreamHandler, FileHandler("scraper.log")]) ejecuta
     → Root logger ahora tiene 2 handlers ← HANDLERS YA CONFIGURADOS
   → log = logging.getLogger(__name__) → logger "whoscored_scraper"

2. run_all_seasons.py continúa (líneas 17-24):
   → logging.basicConfig(handlers=[StreamHandler, FileHandler("scraper_all_seasons.log")])
     → ⚠️ NO-OP: el root logger ya tiene handlers, esta llamada se ignora por completo
   → log = logging.getLogger(__name__) → logger "run_all_seasons"
```

**Resultado**: `scraper_all_seasons.log` **nunca se crea**. Todo el output de
`run_all_seasons.py` va a `scraper.log` (el handler de whoscored_scraper) y a
stdout. El fichero de log específico del run multi-temporada no existe.

### Código afectado

`whoscored_scraper.py` líneas 74-82:
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
```

`run_all_seasons.py` líneas 17-25:
```python
logging.basicConfig(      # ← no-op porque whoscored_scraper ya configuró el root
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scraper_all_seasons.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
```

### Comportamiento al ejecutar whoscored_scraper.py directamente

Cuando se lanza `python whoscored_scraper.py` (sin importar por run_all_seasons),
la configuración de `whoscored_scraper.py` sí funciona y `scraper.log` se crea
correctamente. El bug solo ocurre al ejecutar `run_all_seasons.py`.

## Affected Areas

- `run_all_seasons.py` líneas 17-24 — llamada `basicConfig` que es no-op
- `whoscored_scraper.py` líneas 74-82 — `basicConfig` a nivel de módulo (causa raíz)

## Approaches

### 1. Mover basicConfig de módulo a función `setup_logging()` en whoscored_scraper
Extraer la configuración de logging de nivel de módulo a una función explícita
`setup_logging(log_file)`. Llamarla solo desde `__main__` o desde `run_scraper`.
`run_all_seasons.py` llama su propio `basicConfig` antes de importar, o llama
`setup_logging("scraper_all_seasons.log")`.
- Pros: solución limpia, estándar Python, cada script controla su propio log file
- Cons: requiere cambiar el punto de llamada en `whoscored_scraper.py`
- Esfuerzo: Bajo

### 2. Usar `logging.getLogger` con handler explícito en run_all_seasons
En lugar de `basicConfig`, añadir un `FileHandler` directamente al root logger
o al logger específico de `run_all_seasons` antes de importar whoscored_scraper.
- Pros: no toca whoscored_scraper
- Cons: el orden de imports sigue siendo frágil; poco idiomático
- Esfuerzo: Bajo

### 3. Añadir `force=True` a basicConfig de run_all_seasons (Python ≥ 3.8)
`logging.basicConfig(force=True, ...)` elimina los handlers existentes y
reconfigura el root logger desde cero.
- Pros: una sola línea de cambio en run_all_seasons.py
- Cons: elimina el handler de `scraper.log`; logs de whoscored_scraper ya no
  van a su fichero propio durante el run multi-temporada; confuso
- Esfuerzo: Mínimo

## Recommendation

**Opción 1**: extraer `setup_logging(log_file: str)` en `whoscored_scraper.py`.
El módulo deja de configurar logging al importarse. Cada punto de entrada
(`__main__` de whoscored_scraper y `run_all_seasons.py`) llama `setup_logging`
con su propio fichero. Es el patrón estándar de Python para librerías/módulos
reutilizables: **los módulos no deben llamar `basicConfig` al importarse**.

## Risks

- Cualquier script que importe `whoscored_scraper` sin llamar `setup_logging`
  no tendrá FileHandler. Los logs solo irán a stdout (StreamHandler del root si
  hay alguno configurado, o ningún output si no hay handlers).
  Mitigación: `__main__` de whoscored_scraper llama `setup_logging` explícitamente.

## Ready for Proposal

**Sí.** Bug claro, solución acotada a 2 ficheros, esfuerzo bajo.
