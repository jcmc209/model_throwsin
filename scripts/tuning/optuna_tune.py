"""
Optuna Hyperparameter Tuning — Throw-In Predictor
==================================================
Búsqueda bayesiana de hiperparámetros de LightGBM minimizando mean_val_MAE
sobre los 4 folds completos de walk-forward CV.

Uso:
  python scripts/tuning/optuna_tune.py              # 60 trials (default)
  python scripts/tuning/optuna_tune.py --trials 30  # más rápido
  python scripts/tuning/optuna_tune.py --trials 100 # más exhaustivo

El estudio se guarda en data/model/optuna_study.db (SQLite) para poder
retomarlo o inspeccionar los resultados con optuna-dashboard.

Al final imprime los mejores hiperparámetros y los compara via bootstrap
con el baseline actual.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Asegurar que el root del proyecto esté en el path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("optuna_tune")

STUDY_DB = "data/model/optuna_study.db"
BEST_PARAMS_PATH = "data/model/optuna_best_params.json"


def objective(trial: optuna.Trial) -> float:
    from model.train import run_experiment_cv
    from model.features import SHAP_SELECTED_FEATURES

    params = {
        "num_leaves": trial.suggest_int("num_leaves", 4, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.35, 0.75),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
    }

    result = run_experiment_cv(
        SHAP_SELECTED_FEATURES,
        label=f"trial_{trial.number}",
        lgb_params_override=params,
    )
    return result["mean_val_mae"]


def main(n_trials: int = 60) -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    Path("data/model").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{STUDY_DB}"

    study = optuna.create_study(
        study_name="lgb_throwins_hp",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    existing = len(study.trials)
    remaining = max(0, n_trials - existing)
    log.info("Trials existentes: %d | nuevos a correr: %d", existing, remaining)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)

    best = study.best_trial
    log.info("=== MEJOR TRIAL #%d ===", best.number)
    log.info("  mean_val_MAE = %.4f", best.value)
    log.info("  Params: %s", best.params)

    # Comparar con baseline via bootstrap
    log.info("\nComparando con baseline via bootstrap...")
    from model.train import run_experiment_cv, compare_experiments_bootstrap
    from model.features import SHAP_SELECTED_FEATURES

    baseline = run_experiment_cv(SHAP_SELECTED_FEATURES, label="baseline")
    best_run = run_experiment_cv(
        SHAP_SELECTED_FEATURES,
        label="optuna_best",
        lgb_params_override=best.params,
    )
    cmp = compare_experiments_bootstrap(baseline, best_run, n_boot=2000)

    log.info("Bootstrap result:")
    log.info("  delta       = %+.4f", cmp["delta_mean"])
    log.info("  IC 95%%     = [%+.4f, %+.4f]", cmp["delta_ci_low"], cmp["delta_ci_high"])
    log.info("  p_value     = %.3f", cmp["p_value"])
    log.info("  significant = %s", cmp["significant"])

    # Guardar mejores params
    output = {
        "best_trial": best.number,
        "best_mean_val_mae": round(best.value, 4),
        "baseline_mean_val_mae": baseline["mean_val_mae"],
        "delta_mean": cmp["delta_mean"],
        "delta_ci_low": cmp["delta_ci_low"],
        "delta_ci_high": cmp["delta_ci_high"],
        "p_value": cmp["p_value"],
        "significant": cmp["significant"],
        "params": best.params,
    }
    Path(BEST_PARAMS_PATH).write_text(json.dumps(output, indent=2))
    log.info("Mejores params guardados en %s", BEST_PARAMS_PATH)

    if cmp["significant"] and cmp["delta_mean"] < 0:
        log.info("\n✓ MEJORA SIGNIFICATIVA. Para aplicar, actualizar CONFIG['lgb_params']"
                 " en model/train.py con los params de %s", BEST_PARAMS_PATH)
    else:
        log.info("\n✗ Sin mejora significativa. CONFIG['lgb_params'] no cambia.")

    # Top 5 trials
    log.info("\n=== TOP 5 TRIALS ===")
    top5 = sorted(study.trials, key=lambda t: t.value if t.value else float("inf"))[:5]
    for t in top5:
        log.info("  #%d  MAE=%.4f  %s", t.number, t.value, t.params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HP tuning para throw-in predictor")
    parser.add_argument("--trials", type=int, default=60, help="Número total de trials")
    args = parser.parse_args()
    main(n_trials=args.trials)
