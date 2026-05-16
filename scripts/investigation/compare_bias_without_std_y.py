"""
compare_bias_without_std_y
==========================
Investigación empírica (followup/leakage-investigation-std-y, engram #27):
entrena un clon del modelo de producción excluyendo `std_y` (y `opp_std_y`)
de la lista de features, y compara los team-bias post-shrinkage contra el
modelo actual — con especial foco en los dos equipos flagged:

    - Athletic Club (team_id=53) AWAY
    - Real Betis    (team_id=54) HOME

Contrato (aislamiento de producción):
  - NO sobrescribe `data/model/model_v1.joblib`.
  - NO sobrescribe `data/model/team_bias_calibration_v2.json`.
  - Artefactos temporales con prefijo `_investigation_*` en `data/model/`.
  - Usa `model.market_utils.shrink_bias` (ADR D2 — fuente única) con los
    mismos hiperparámetros que el modelo de producción (k=5.0, prior_mu=0).
  - Mismo split temporal (`CONFIG["train_seasons"]` vs `CONFIG["val_seasons"]`).

Uso:
    python -m scripts.investigation.compare_bias_without_std_y

Salidas:
    data/model/_investigation_no_std_y.joblib
    data/model/_investigation_team_bias_no_std_y.json
    data/model/_investigation_comparison_no_std_y.csv
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model.features import SHAP_SELECTED_FEATURES, TARGET_COL
from model.market_utils import shrink_bias
from model.train import CONFIG as TRAIN_CFG
from model.train import (
    baseline_team_mean,
    build_sample_weights,
    load_dataset,
    split_temporal,
    train_lightgbm,
)

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("investigate_std_y")

# ─────────────────────────────────────────────────────────────
# CONFIG LOCAL (paths tmp, NUNCA pisa producción)
# ─────────────────────────────────────────────────────────────
INV_CFG = {
    "model_tmp_path": "data/model/_investigation_no_std_y.joblib",
    "team_bias_tmp_path": "data/model/_investigation_team_bias_no_std_y.json",
    "comparison_csv_path": "data/model/_investigation_comparison_no_std_y.csv",
    "prod_team_bias_path": TRAIN_CFG["team_bias_path"],
    "features_to_drop": ["std_y", "opp_std_y"],
    "shrinkage_k": TRAIN_CFG["team_bias_shrinkage_k"],
    "prior_mu": TRAIN_CFG["team_bias_prior_mu"],
    # Cells bajo investigación — los flagged por discovery #22
    "spotlight_cells": [(53, 0), (54, 1)],
}


def _compute_shrunk_biases(
    model: lgb.LGBMRegressor,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    k: float,
    prior_mu: float,
) -> dict[tuple[int, int], dict[str, float]]:
    """
    Para cada (team_id, is_home) en `val_df`, devuelve {n, raw_bias, shrunk_bias}.

    Usa `market_utils.shrink_bias` (fuente única ADR D2).
    """
    X_val = val_df[feature_cols].astype(float)
    y_true = val_df[TARGET_COL].astype(float).to_numpy()
    lam_pred = np.asarray(model.predict(X_val), dtype=float)
    residuals = y_true - lam_pred

    df = pd.DataFrame({
        "team_id": val_df["team_id"].to_numpy(),
        "is_home": val_df["is_home"].to_numpy(),
        "residual": residuals,
    })
    grouped = df.groupby(["team_id", "is_home"])["residual"].agg(["count", "mean"])

    out: dict[tuple[int, int], dict[str, float]] = {}
    for (tid, hid), row in grouped.iterrows():
        n = int(row["count"])
        raw = float(row["mean"])
        shrunk = shrink_bias(raw, n, k=k, prior_mu=prior_mu)
        out[(int(tid), int(hid))] = {
            "n": n,
            "raw_bias": round(raw, 4),
            "shrunk_bias": round(shrunk, 4),
        }
    return out


def _load_prod_shrunk_biases(prod_json_path: Path) -> dict[tuple[int, int], dict[str, float]]:
    """Lee `team_bias_calibration_v2.json` y lo vuelca al formato {(tid,hid): {...}}."""
    with open(prod_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    corrections = payload.get("corrections", {})
    out: dict[tuple[int, int], dict[str, float]] = {}
    for tid_str, by_home in corrections.items():
        for hid_str, stats in by_home.items():
            out[(int(tid_str), int(hid_str))] = {
                "n": int(stats["n"]),
                "raw_bias": float(stats["raw_bias"]),
                "shrunk_bias": float(stats["shrunk_bias"]),
            }
    return out


def _build_team_name_lookup(val_df: pd.DataFrame) -> dict[int, str]:
    """Mapa team_id → team_name usando el val DataFrame."""
    pairs = val_df[["team_id", "team_name"]].drop_duplicates()
    return {int(r.team_id): str(r.team_name) for r in pairs.itertuples(index=False)}


def train_without_std_y() -> tuple[lgb.LGBMRegressor, pd.DataFrame, list[str], str]:
    """
    Replica `model.train.main(features_mode='shap30')` pero con std_y / opp_std_y
    excluidas de feature_cols. Devuelve (modelo, val_df, feature_cols, trained_at).

    No escribe métricas ni JSON a producción — solo el joblib tmp.
    """
    df = load_dataset()
    train, val, _test = split_temporal(df)

    baseline_mae = baseline_team_mean(train, val)
    log.info("Baseline team-mean MAE (val) = %.4f", baseline_mae)

    # Feature list de producción — mismo conjunto SHAP30 — MENOS las sospechosas.
    drop_set = set(INV_CFG["features_to_drop"])
    feature_cols = [c for c in SHAP_SELECTED_FEATURES if c not in drop_set]
    dropped = [c for c in SHAP_SELECTED_FEATURES if c in drop_set]
    log.info(
        "Features originales=%d, descartadas=%d (%s), quedan=%d",
        len(SHAP_SELECTED_FEATURES), len(dropped), dropped, len(feature_cols),
    )

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        raise ValueError(f"features ausentes en dataset: {missing}")

    X_train = train[feature_cols]
    y_train = train[TARGET_COL]
    X_val = val[feature_cols]
    y_val = val[TARGET_COL]

    # Mismo esquema de selección que producción: prueba uniform+decay, elige menor val_MAE.
    from sklearn.metrics import mean_absolute_error

    best_model = None
    best_mae = float("inf")
    best_scheme = None
    for scheme in ("uniform", "decay"):
        sw = build_sample_weights(train, scheme)
        model = train_lightgbm(X_train, y_train, X_val, y_val, sw)
        val_mae = float(mean_absolute_error(y_val, model.predict(X_val)))
        log.info("  weights=%s → val_MAE %.4f (best_iter=%d)",
                 scheme, val_mae, model.best_iteration_ or model.n_estimators)
        if val_mae < best_mae:
            best_mae = val_mae
            best_model = model
            best_scheme = scheme

    log.info("Mejor esquema (sin std_y): %s (val_MAE=%.4f)", best_scheme, best_mae)

    trained_at = datetime.utcnow().isoformat(timespec="seconds")
    return best_model, val, feature_cols, trained_at


def write_tmp_artifacts(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    trained_at: str,
    biases: dict[tuple[int, int], dict[str, float]],
) -> None:
    """Guarda joblib tmp y JSON de biases tmp (NUNCA toca paths de producción)."""
    # Joblib tmp
    joblib_path = Path(INV_CFG["model_tmp_path"])
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "version": "investigation_no_std_y",
        "trained_at": trained_at,
        "features": feature_cols,
        "params": dict(TRAIN_CFG["lgb_params"]),
        "train_seasons": TRAIN_CFG["train_seasons"],
        "val_seasons": TRAIN_CFG["val_seasons"],
        "features_dropped": INV_CFG["features_to_drop"],
    }
    joblib.dump(artifact, joblib_path)
    log.info("joblib tmp guardado: %s", joblib_path)

    # JSON tmp (mismo schema que prod, pero marcado como investigación)
    corrections: dict[str, dict[str, dict[str, float]]] = {}
    for (tid, hid), stats in sorted(biases.items()):
        corrections.setdefault(str(tid), {})[str(hid)] = {
            "n": stats["n"],
            "raw_bias": stats["raw_bias"],
            "shrunk_bias": stats["shrunk_bias"],
        }
    payload = {
        "description": (
            "INVESTIGACIÓN — biases post-shrinkage sin std_y/opp_std_y "
            "(followup/leakage-investigation-std-y). NO ES PRODUCCIÓN."
        ),
        "generated_at": datetime.utcnow().isoformat() + "+00:00",
        "model_trained_at": trained_at,
        "model_train_seasons": list(TRAIN_CFG["train_seasons"]),
        "shrinkage_k": INV_CFG["shrinkage_k"],
        "prior_mu": INV_CFG["prior_mu"],
        "features_dropped": INV_CFG["features_to_drop"],
        "corrections": corrections,
    }
    json_path = Path(INV_CFG["team_bias_tmp_path"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    log.info("JSON biases tmp guardado: %s", json_path)


def build_comparison_table(
    biases_with: dict[tuple[int, int], dict[str, float]],
    biases_without: dict[tuple[int, int], dict[str, float]],
    name_lookup: dict[int, str],
) -> pd.DataFrame:
    """
    Tabla por (team_id, is_home) con biases WITH vs WITHOUT std_y.

    Columnas:
        team_id, team_name, side, n, raw_with, raw_without, delta_raw,
        shrunk_with, shrunk_without, delta_shrunk.

    `side` ∈ {"home", "away"}.
    """
    # Universo de cells = unión, aunque normalmente son las mismas 40.
    cells = sorted(set(biases_with.keys()) | set(biases_without.keys()))
    rows: list[dict] = []
    for tid, hid in cells:
        w = biases_with.get((tid, hid), {})
        wo = biases_without.get((tid, hid), {})
        row = {
            "team_id": tid,
            "team_name": name_lookup.get(tid, f"team_{tid}"),
            "side": "home" if hid == 1 else "away",
            "n": wo.get("n") or w.get("n") or 0,
            "raw_with": w.get("raw_bias", np.nan),
            "raw_without": wo.get("raw_bias", np.nan),
            "delta_raw": (
                round(float(wo["raw_bias"] - w["raw_bias"]), 4)
                if "raw_bias" in w and "raw_bias" in wo
                else np.nan
            ),
            "shrunk_with": w.get("shrunk_bias", np.nan),
            "shrunk_without": wo.get("shrunk_bias", np.nan),
            "delta_shrunk": (
                round(float(wo["shrunk_bias"] - w["shrunk_bias"]), 4)
                if "shrunk_bias" in w and "shrunk_bias" in wo
                else np.nan
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["team_id", "side"]
    ).reset_index(drop=True)


def emit_verdict(table: pd.DataFrame) -> dict:
    """
    Regla de decisión sobre las cells spotlight:
      - |shrunk_without| < 1 → leakage CONFIRMED.
      - |delta_shrunk| < 0.3 → leakage REFUTED.
      - en medio → AMBIGUOUS.

    Devuelve dict con campos `verdict`, `details_per_cell`, y `overall_mean_abs_shrunk`.
    """
    details: list[dict] = []
    for tid, hid in INV_CFG["spotlight_cells"]:
        side = "home" if hid == 1 else "away"
        row = table[(table["team_id"] == tid) & (table["side"] == side)]
        if row.empty:
            details.append({
                "team_id": tid, "side": side, "status": "MISSING",
            })
            continue
        r = row.iloc[0]
        abs_with = abs(float(r["shrunk_with"]))
        abs_without = abs(float(r["shrunk_without"]))
        delta = float(r["delta_shrunk"])
        if abs_without < 1.0:
            cell_verdict = "CONFIRMED"
        elif abs(delta) < 0.3:
            cell_verdict = "REFUTED"
        else:
            cell_verdict = "AMBIGUOUS"
        details.append({
            "team_id": tid,
            "team_name": str(r["team_name"]),
            "side": side,
            "n": int(r["n"]),
            "shrunk_with": round(abs_with, 4) if np.isnan(r["shrunk_with"]) is False else None,
            "shrunk_without": round(abs_without, 4),
            "delta_shrunk": round(delta, 4),
            "cell_verdict": cell_verdict,
        })

    # Verdict global: si TODOS spotlight cells dan CONFIRMED → CONFIRMED.
    # Si TODOS dan REFUTED → REFUTED. Mezclado → AMBIGUOUS.
    cell_verdicts = {d["cell_verdict"] for d in details if "cell_verdict" in d}
    if cell_verdicts == {"CONFIRMED"}:
        overall = "CONFIRMED"
    elif cell_verdicts == {"REFUTED"}:
        overall = "REFUTED"
    else:
        overall = "AMBIGUOUS"

    overall_mean = float(np.nanmean(np.abs(table["shrunk_without"].to_numpy(dtype=float))))
    return {
        "verdict": overall,
        "details_per_cell": details,
        "overall_mean_abs_shrunk_without": round(overall_mean, 4),
        "overall_mean_abs_shrunk_with": round(
            float(np.nanmean(np.abs(table["shrunk_with"].to_numpy(dtype=float)))), 4
        ),
    }


def main() -> int:
    log.info("=== LEG 2 — Retraining without std_y / opp_std_y ===")
    model, val_df, feature_cols, trained_at = train_without_std_y()

    log.info("Computando biases post-shrinkage (sin std_y)...")
    biases_without = _compute_shrunk_biases(
        model,
        val_df,
        feature_cols,
        INV_CFG["shrinkage_k"],
        INV_CFG["prior_mu"],
    )
    log.info("Biases computados: %d cells (team×side)", len(biases_without))

    log.info("Cargando biases de PRODUCCIÓN (con std_y)...")
    biases_with = _load_prod_shrunk_biases(Path(INV_CFG["prod_team_bias_path"]))
    log.info("Biases prod: %d cells", len(biases_with))

    name_lookup = _build_team_name_lookup(val_df)
    table = build_comparison_table(biases_with, biases_without, name_lookup)
    log.info("Tabla comparativa: %d filas", len(table))

    # Persistir artefactos tmp
    write_tmp_artifacts(model, feature_cols, trained_at, biases_without)

    csv_path = Path(INV_CFG["comparison_csv_path"])
    table.to_csv(csv_path, index=False)
    log.info("CSV comparación guardado: %s", csv_path)

    # Spotlight
    print("\n" + "=" * 80)
    print("SPOTLIGHT — flagged cells (followup/leakage-investigation-std-y)")
    print("=" * 80)
    for tid, hid in INV_CFG["spotlight_cells"]:
        side = "home" if hid == 1 else "away"
        r = table[(table["team_id"] == tid) & (table["side"] == side)]
        if r.empty:
            print(f"  team_id={tid} {side}: NOT FOUND in val")
            continue
        r = r.iloc[0]
        print(
            f"  team_id={tid} ({r['team_name']}) {side.upper()} (n={int(r['n'])})\n"
            f"    WITH std_y    : raw_bias={r['raw_with']:+.4f}  shrunk_bias={r['shrunk_with']:+.4f}\n"
            f"    WITHOUT std_y : raw_bias={r['raw_without']:+.4f}  shrunk_bias={r['shrunk_without']:+.4f}\n"
            f"    Δ_shrunk      : {r['delta_shrunk']:+.4f}"
        )

    verdict = emit_verdict(table)
    print("\n" + "=" * 80)
    print(f"VERDICT: {verdict['verdict']}")
    print("=" * 80)
    print(
        f"  mean|shrunk| WITH    std_y = {verdict['overall_mean_abs_shrunk_with']:.4f}\n"
        f"  mean|shrunk| WITHOUT std_y = {verdict['overall_mean_abs_shrunk_without']:.4f}"
    )
    for d in verdict["details_per_cell"]:
        if d.get("status") == "MISSING":
            print(f"  team_id={d['team_id']} {d['side']}: MISSING")
            continue
        print(
            f"  [{d['cell_verdict']:9s}] {d['team_name']:20s} {d['side']:5s} "
            f"n={d['n']:<3d} |shrunk_with|={d['shrunk_with']} "
            f"|shrunk_without|={d['shrunk_without']} Δ={d['delta_shrunk']:+.4f}"
        )

    # Además, top-15 |shrunk_with| ordenados — muestra el efecto general.
    print("\nTop-15 cells por |shrunk_with| (ordenadas desc.):")
    top = table.assign(
        abs_w=table["shrunk_with"].abs()
    ).sort_values("abs_w", ascending=False).head(15)
    print(
        top[[
            "team_id", "team_name", "side", "n",
            "raw_with", "raw_without", "delta_raw",
            "shrunk_with", "shrunk_without", "delta_shrunk",
        ]].to_string(index=False)
    )

    print(f"\nCSV:    {csv_path}")
    print(f"joblib: {INV_CFG['model_tmp_path']}")
    print(f"JSON:   {INV_CFG['team_bias_tmp_path']}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
