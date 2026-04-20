# Exploration: entorno-reproducible

## Current State

El proyecto no tiene ningún fichero formal de dependencias. Las dependencias reales
se detectan leyendo los imports de `whoscored_scraper.py`:

| Paquete pip | Import en código | ¿Obligatorio? |
|-------------|-----------------|--------------|
| `pandas` | `import pandas as pd` | Sí |
| `tqdm` | `from tqdm import tqdm` | Sí |
| `playwright` | `from playwright.sync_api import ...` | Sí |
| `python-dotenv` | `from dotenv import load_dotenv` | Sí |
| `pymongo` | `from pymongo import ...` | Sí |
| `pyarrow` | implícito en `.to_parquet()` | Sí (sin él parquet falla silenciosamente) |
| `mplsoccer` | `from mplsoccer import Pitch` (lazy, en try/except) | No |
| `matplotlib` | `import matplotlib.pyplot as plt` (lazy, en try/except) | No |

Variables de entorno que el código referencia (`os.getenv`):
- `MONGO_URI` (línea 144) — **obligatoria**; lanza `ValueError` si no existe
- `MONGO_DB` (línea 145) — opcional, por defecto `"modelthrowins"`

Estado actual de ficheros de entorno en la raíz:
- `requirements.txt` — **ya existe** (creado en sesión anterior, contenido correcto)
- `.env.example` — **ya existe** (creado en sesión anterior, contenido correcto)
- `.env` — **ya existe** (fichero real del usuario, no tocar)
- Sin `pyproject.toml`, `Pipfile` ni `setup.py`

## Affected Areas

- `whoscored_scraper.py` — fuente de verdad de las dependencias reales (no se modifica)
- `requirements.txt` — ya creado; revisar si está completo
- `.env.example` — ya creado; revisar si cubre todas las variables
- `test_single.py` y `validate.py` — se benefician del entorno documentado (no se modifican)

## Approaches

### 1. Validar y cerrar (estado actual es suficiente)
Los ficheros creados en la sesión anterior son correctos y completos.
Verificar cobertura y archivar el cambio.
- Pros: mínimo esfuerzo, no hay deuda técnica real pendiente
- Cons: ninguno
- Esfuerzo: Bajo

### 2. Añadir `pyproject.toml` además de `requirements.txt`
Estructura más moderna; permite `optional-dependencies` formales.
- Pros: estándar PEP 517/518, permite `pip install ".[viz]"`
- Cons: sobreingeniería para un repo de scripts sin empaquetado
- Esfuerzo: Medio

## Recommendation

**Opción 1**: los ficheros `requirements.txt` y `.env.example` ya existentes cubren
todos los requisitos. El cambio consiste en validarlos formalmente contra el código
y archivarlos bajo SDD. No se necesita `pyproject.toml` a este nivel de madurez.

## Risks

- `pyarrow` no aparece como import directo pero es necesario para `.to_parquet()`;
  si se omite, el fallo es silencioso (pandas lanza error en runtime). Ya está en
  `requirements.txt`, sin riesgo.
- `.env` real ya existe en disco; `.env.example` no lo sobreescribe. Sin riesgo.

## Ready for Proposal

**Sí.** Alcance claro, implementación ya presente, solo falta validación formal y
cierre SDD del cambio.
