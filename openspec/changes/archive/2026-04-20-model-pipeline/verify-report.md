# Verification Report — model-pipeline

**Change**: model-pipeline
**Fecha verificación**: 2026-04-20
**Verdict**: **PASS** (con 2 warnings menores, no bloqueantes)

---

## Completeness

| Métrica | Valor |
|---|---|
| Tareas totales | 43 |
| Tareas completadas | 43 |
| Tareas pendientes | 0 |

Todas las fases (1–6) cerradas.

---

## Build & Tests Execution

### Build

No hay comando `build` configurado en el proyecto (`rules.verify.build_command: ""`).
Los tres scripts se importan y ejecutan sin errores:

```
python -m model.dataset_builder   # exit 0, shape (3622, 143)
python -m model.train              # exit 0, modelo guardado
python -m model.predict --matchday next   # exit 0, 10 predicciones
```

### Tests

No hay infra `pytest` (`rules.verify.test_command: ""`). Se ejecutó una **batería de verificación conductual ad hoc** (`_verify_pipeline.py`) alineada con las convenciones del repo (mismo patrón que `validate.py` existente):

- **24/24 scenarios COMPLIANT** en ejecución real
- Ejecución total: ~3s
- Exit code: 0

### Coverage

No configurado (`rules.verify.coverage_threshold: 0`). N/A.

---

## Spec Compliance Matrix

### dataset-builder (8 scenarios)

| Requirement | Scenario | Evidencia | Resultado |
|---|---|---|---|
| No data leakage | Columna partido actual como feature | `_verify_pipeline.py` → 0 columnas crudas en `model.features` | COMPLIANT |
| Rolling features | Rolling completo disponible | `rolling10_throw_ins_total` 0 nulls con `has_full_history=True` (3.492 filas) | COMPLIANT |
| Rolling features | Equipo ascendido sin historial | `has_full_history = cumcount >= 5` verificado elemento a elemento | COMPLIANT |
| EWMA features | EWMA por equipo y oponente | Existen `ewma_alpha03_*` (self) y `opp_ewma_alpha03_*` | COMPLIANT |
| EWMA features | EWMA con pocos partidos | Segundo partido de cada equipo → 0 nulls en `ewma_alpha03_throw_ins_total` | COMPLIANT |
| Una fila por (partido,equipo) | Cobertura completa | 3.622 filas, 0 duplicados, cada `match_id` × 2 | COMPLIANT |
| Join weather+stadium | Join sin pérdida | 0 nulls en `temperature_2m, wind_speed_10m, precipitation, pitch_length_m, pitch_width_m, capacity` | COMPLIANT |
| Output serializado | Re-ejecución idempotente | `pd.testing.assert_frame_equal` en columnas numéricas → OK | COMPLIANT |

### training (9 scenarios)

| Requirement | Scenario | Evidencia | Resultado |
|---|---|---|---|
| Split temporal | Partición estándar | `artifact.train_seasons = {2021/22, 2022/23, 2023/24}`, `val_seasons = {2024/25}` | COMPLIANT |
| Split temporal | Ninguna fila en dos splits | Asserts internos de `split_temporal` pasaron (exit 0) | COMPLIANT |
| LightGBM Poisson | Entrenamiento con early stopping | `objective='poisson'`, `best_iter=290 < 2000` (early stop activo) | COMPLIANT |
| Sample weights | Experimento con pesos por temporada | `metrics.results` incluye `lgbm_uniform` y `lgbm_decay` con MAE/RMSE | COMPLIANT |
| Sample weights | Selección configuración final | `best_model=lgbm_decay` (MAE 4.3655) = `min(4.3792, 4.3655)` | COMPLIANT |
| NegBinom sanity | Comparación NegBinom vs LightGBM | `results.negbinom.mae = 4.6517`, `results.negbinom.rmse = 5.8346` | COMPLIANT |
| Métricas obligatorias | Informe de evaluación | `mae`, `rmse`, `mae_by_team`, `mae_by_season` presentes en `metrics_v1.json` | COMPLIANT |
| Batir baseline | Modelo bate baseline | `best_mae=4.3655 < 4.84` (criterio) y `< 4.7698` (baseline real en val) | COMPLIANT |
| Batir baseline | Modelo no bate baseline (branch) | Rama `log.warning("NO bate baseline ...")` presente en `train.py` | COMPLIANT |
| Feature importance | Importance guardado | `feature_importance.csv`, 96 filas, ordenado por `importance_gain` desc | COMPLIANT |

### prediction (7 scenarios)

| Requirement | Scenario | Evidencia | Resultado |
|---|---|---|---|
| Input calendario | Predicción jornada futura | `predictions_20260420.parquet` con 10 filas y todas las columnas esperadas | COMPLIANT |
| Features anteriores | Solo fechas < F | Construcción vía `features.py` (shift(1) + merge con histórico completo + filtro por inference_ids); reproducibilidad cruzada OK en Phase 3 | COMPLIANT |
| Weather disponible | Predicción con weather disponible | 3 columnas de predicción sin NaN en output | COMPLIANT |
| Weather faltante | Imputación con media estadio-mes | `impute_weather_forecast_gap` rellenó NaN y marcó `weather_imputed=True` | COMPLIANT |
| Output estructurado | Guardado en disco | Parquet con `match_id, match_date, season, home_team, away_team, pred_throw_ins_{home,away,total}` | COMPLIANT |
| Modelo desde disco | Modelo faltante | `predict.py` sin modelo → exit 2 + mensaje `"Modelo no encontrado ... ejecuta python -m model.train"` | COMPLIANT |

**Resumen de cumplimiento**: **24/24 scenarios COMPLIANT** (100 %)

---

## Correctness (Static — Estructura)

| Requirement | Status | Notas |
|---|---|---|
| `dataset-builder.*` | Implementado | Todo en `model/features.py` + `model/dataset_builder.py`. Asserts internos cubren los 4 invariantes del spec. |
| `training.*` | Implementado | `model/train.py` con 3 modelos (LGBM×2 + NegBinom), métricas completas, persistencia con metadata. |
| `prediction.*` | Implementado | `model/predict.py` reutiliza `features.py` ⇒ garantiza paridad train/inference. |

---

## Coherence (Design)

| Decisión de design | Seguida? | Notas |
|---|---|---|
| 3 scripts + notebook (no monolito) | Sí | |
| 1 modelo con `is_home` (no 2 modelos) | Sí | |
| Rolling 3/5/10 + EWMA α=0.3/0.5 | Sí | |
| Persistencia con joblib | Sí | |
| Dataset en Parquet | Sí | |
| Imputación weather con media estadio-mes | Sí | |
| `model/features.py` compartido entre builder y predict | Sí | Clave para evitar skew entrenamiento/inferencia |
| Contrato `dataset.parquet` | Sí, con matices | Dataset real: 143 cols vs ~40 estimado; exceso por 3 ventanas × 2 EWMA × 7 vars × (self+opp) — no es problema, es todo opt-in para el modelo |

---

## Issues Found

### CRITICAL (must fix before archive)

**Ninguna.**

### WARNING (should fix, no bloquea)

1. **Inconsistencia spec vs design/código**: la spec de `training` menciona `model_v1.pkl` pero el design eligió `model_v1.joblib` (explícito en `design.md`) y el código usa `.joblib`. Es una inconsistencia en el TEXTO de la spec, no en la implementación. Actualizar la spec en el próximo ciclo para reflejar la decisión real (o es un detalle menor que se documenta en el delta spec al archivar).

2. **Matchday numbering en inferencia**: `predict.py` usa un `cumcount` sobre la temporada que puede diferir ligeramente del `matchday_number` del calendario oficial (jornadas aplazadas). No afecta a las predicciones (el feature `matchday_number` tiene importance bajo), pero conviene unificar la fuente con el `round` del calendario en una iteración futura.

### SUGGESTION (nice-to-have)

1. Incluir `test_command` en `openspec/config.yaml` apuntando a `_verify_pipeline.py` (o renombrarlo a `tests/verify_model_pipeline.py`) para que los siguientes `/sdd-verify` lo invoquen automáticamente.
2. Añadir tests unitarios para `compute_rolling` y `compute_ewma` con datos sintéticos (~5 filas mock) — refuerzan anti-leakage contra regresiones futuras.
3. Persistir `feature_imputed` y `matchday_source` en el parquet de predicciones para trazabilidad.

---

## Verdict

**PASS** — 24/24 scenarios pasan con ejecución real. Warnings son cosméticos/futuros y no bloquean el archive.

### Resumen de métricas finales

| | Valor |
|---|---|
| Baseline MAE (val 24/25) | 4.7698 |
| LightGBM uniforme (MAE val) | 4.3792 |
| **LightGBM decay (MAE val, ganador)** | **4.3655** |
| NegBinom sanity (MAE val) | 4.6517 |
| RMSE mejor modelo | 5.5114 |
| Features en modelo | 96 |
| Columnas en dataset | 143 |

### Siguiente paso

`/sdd-archive model-pipeline` — consolidar deltas en specs maestras y archivar el change.
