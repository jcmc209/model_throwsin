# Archive Report — model-pipeline

**Archivado**: 2026-04-20
**Carpeta**: `openspec/changes/archive/2026-04-20-model-pipeline/`
**Verificación previa**: PASS (24/24 scenarios, 0 issues críticos)

## Specs sincronizadas a main

| Dominio | Acción | Detalles |
|---|---|---|
| `dataset-builder` | Creada | 5 requirements, 8 scenarios (dominio nuevo) |
| `training` | Creada | 6 requirements, 9 scenarios (dominio nuevo) — corregido `.pkl` → `.joblib` |
| `prediction` | Creada | 5 requirements, 7 scenarios (dominio nuevo) |

Specs master en:
- `openspec/specs/dataset-builder/spec.md`
- `openspec/specs/training/spec.md`
- `openspec/specs/prediction/spec.md`

## Contenido archivado

- `exploration.md` ✅
- `proposal.md` ✅
- `design.md` ✅
- `specs/` ✅ (dataset-builder, training, prediction)
- `tasks.md` ✅ (43/43 tasks complete)
- `verify-report.md` ✅ (PASS, 24/24 scenarios)
- `archive-report.md` ✅ (este fichero)

## Artefactos de código generados

| Artefacto | Ruta |
|---|---|
| Feature engineering | `model/features.py` |
| Dataset builder | `model/dataset_builder.py` |
| Entrenamiento | `model/train.py` |
| Predicción | `model/predict.py` |
| Notebook EDA | `notebooks/01_eda.ipynb` |
| Dataset | `data/model/dataset.parquet` |
| Modelo serializado | `data/model/model_v1.joblib` |
| Métricas | `data/model/metrics_v1.json` |
| Feature importance | `data/model/feature_importance.csv` |
| Predicciones | `data/model/predictions_20260420.parquet` |

## Resultados finales

| Modelo | MAE val 2024/25 | RMSE |
|---|---|---|
| Baseline (media por equipo) | 4.7698 | — |
| LightGBM Poisson uniforme | 4.3792 | 5.5000 |
| **LightGBM Poisson decay** | **4.3655** | 5.5114 |
| NegBinom sanity check | 4.6517 | 5.8346 |

Criterio de éxito original (MAE < 4.84): **cumplido** (4.3655).

## Warnings documentados (no bloquearon el archive)

1. Spec `training` mencionaba `model_v1.pkl`; corregido a `.joblib` en la spec master.
2. `matchday_number` en inferencia usa `cumcount` en vez del `round` del calendario oficial — efecto despreciable (importance bajo), a resolver en iteración futura.

## Siguiente cambio sugerido

`betting-evaluator` — recolección de cuotas Codere + integración de datos de throws por partido para evaluar el edge del modelo contra el mercado.
