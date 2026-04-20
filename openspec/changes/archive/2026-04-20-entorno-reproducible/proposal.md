# Proposal: entorno-reproducible

## Intent

Cualquier persona que clone el repo debe poder reproducir el entorno con dos
comandos y saber qué variables de entorno configurar, sin leer el código fuente.
Actualmente no existe ningún fichero de dependencias ni de entorno documentado.

## Scope

### In Scope
- `requirements.txt` con todas las dependencias (core + opcionales marcadas)
- `.env.example` con todas las variables de entorno documentadas y comentadas
- Nota post-install para `playwright install chromium`

### Out of Scope
- `pyproject.toml` / empaquetado formal
- CI/CD, Docker, pinning exacto de versiones
- Modificaciones a ningún fichero `.py`

## Approach

Crear dos ficheros en la raíz. Los ficheros ya existen de una sesión anterior
con contenido correcto; este ciclo los valida formalmente contra el código y
los formaliza bajo SDD.

## Affected Areas

| Área | Impacto | Descripción |
|------|---------|-------------|
| `requirements.txt` | Nuevo | 6 deps core + 2 opcionales + nota playwright |
| `.env.example` | Nuevo | `MONGO_URI` (obligatoria) + `MONGO_DB` (opcional) |
| `whoscored_scraper.py` | Sin cambios | Fuente de verdad de imports y `os.getenv` |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|-----------|
| `pyarrow` no es import directo, fallo silencioso si se omite | Baja | Ya incluido en `requirements.txt` |
| `.env` real sobreescrito | Baja | `.env.example` es solo plantilla; `.env` no se toca |

## Rollback Plan

Borrar `requirements.txt` y `.env.example`. Sin impacto en código ni datos.

## Dependencies

Ninguna externa.

## Success Criteria

- [ ] `pip install -r requirements.txt` no lanza errores en entorno limpio
- [ ] `playwright install chromium` está documentado en el fichero
- [ ] Todas las variables de `os.getenv` en `whoscored_scraper.py` aparecen en `.env.example`
- [ ] Ningún fichero `.py` modificado
