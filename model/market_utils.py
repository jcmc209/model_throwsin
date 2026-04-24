"""
Market Utilities — Throw-In Predictor
======================================
Funciones puras para el pipeline de evaluación vs mercado y value betting.

Importar desde scripts/evaluation/:
  from model.market_utils import devig_proportional, poisson_over_prob, TEAM_NAME_MAP

Funciones disponibles:
  devig_proportional(odds_a, odds_b)  → (p_a, p_b)
  devig_shin(odds_a, odds_b)          → (p_a, p_b)
  poisson_over_prob(lam_total, line)  → float
  poisson_under_prob(lam_total, line) → float
  nbinom_over_prob(mu, line, alpha)   → float | ndarray  (tail con dispersion α)
  nbinom_under_prob(mu, line, alpha)  → float | ndarray
  normalize_team(name)                → str
  load_team_bias(path)                → dict
  apply_team_bias(lam, team_id, is_home, bias_table)  → ndarray
  shrink_bias(raw_bias, n, k, prior_mu) → float   (Bayesian normal-normal)
  p_home_more(lam_h, lam_w, method)   → (p_home, p_tie, p_away)

Constantes:
  DEFAULT_NEGBIN_ALPHA — α tuneado offline (grid_mle) sobre val 2025/2026.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import nbinom, poisson, skellam

log = logging.getLogger("market_utils")

_DEFAULT_BIAS_PATH = "data/model/team_bias_calibration_v2.json"
_MISSING_BIAS_FILE_WARNED: set[str] = set()
_MISSING_TEAM_WARNED: set[tuple[str, str]] = set()

# ─────────────────────────────────────────────────────────────────────────────
# MAPEO DE NOMBRES  Codere → WhoScored (dataset.parquet)
# Añadir nuevas entradas cuando Codere incorpore equipos con nombre diferente.
# ─────────────────────────────────────────────────────────────────────────────

TEAM_NAME_MAP: dict[str, str] = {
    # Codere short   →   WhoScored full
    "Alavés":           "Deportivo Alaves",
    "Athletic":         "Athletic Club",
    "Atlético":         "Atletico",
    "Celta":            "Celta Vigo",
    "Oviedo":           "Real Oviedo",
    "Rayo":             "Rayo Vallecano",
    "Levante UD":       "Levante",
    # Nombres que ya coinciden (identity mappings — opcionales pero explícitos)
    "Barcelona":        "Barcelona",
    "Real Madrid":      "Real Madrid",
    "Getafe":           "Getafe",
    "Girona":           "Girona",
    "Mallorca":         "Mallorca",
    "Osasuna":          "Osasuna",
    "Villarreal":       "Villarreal",
    "Espanyol":         "Espanyol",
    "Real Betis":       "Real Betis",
    "Real Sociedad":    "Real Sociedad",
    "Sevilla":          "Sevilla",
    "Valencia":         "Valencia",
    "Elche":            "Elche",
}

# Inverso: WhoScored → Codere (útil para cruzar predicciones → cuotas)
_DS_TO_CODERE: dict[str, str] = {v: k for k, v in TEAM_NAME_MAP.items()}


def normalize_team(name: str, direction: str = "codere_to_ds") -> str:
    """
    Normaliza nombre de equipo entre fuentes.
    direction='codere_to_ds' (default): Codere → dataset
    direction='ds_to_codere': dataset → Codere
    Devuelve el nombre original si no hay mapeo conocido.
    """
    if direction == "codere_to_ds":
        return TEAM_NAME_MAP.get(name, name)
    return _DS_TO_CODERE.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# DEVIG
# ─────────────────────────────────────────────────────────────────────────────

def devig_proportional(odds_a: float, odds_b: float) -> tuple[float, float]:
    """
    Devig proporcional (método estándar, market-neutral).
    Elimina el margen de la casa distribuyéndolo proporcionalmente.

    Args:
        odds_a: cuota del lado A (e.g. over)
        odds_b: cuota del lado B (e.g. under)

    Returns:
        (p_a, p_b) — probabilidades devigged que suman 1.0
    """
    raw_a = 1.0 / odds_a
    raw_b = 1.0 / odds_b
    total = raw_a + raw_b
    return raw_a / total, raw_b / total


def devig_shin(odds_a: float, odds_b: float) -> tuple[float, float]:
    """
    Método de Shin (1993) — asume insider trading asimétrico.
    Más preciso en mercados con apostantes informativos; no cambia mucho
    en mercados thin como saques de banda.

    Returns:
        (p_a, p_b) — probabilidades devigged que suman 1.0
    """
    p_raw_a = 1.0 / odds_a
    p_raw_b = 1.0 / odds_b
    vig = p_raw_a + p_raw_b - 1.0  # exceso sobre 1

    # Shin: resolver ecuación cuadrática
    # z = fracción de apostantes con información privada
    # p_shin_i ≈ sqrt(z^2 + 4*(1-z)*p_raw_i/vig) - z) / (2*(1-z))
    # Aproximación práctica iterativa de dos iteraciones
    z = vig / (2 * vig + 2)
    def _shin_one(p_raw: float) -> float:
        disc = z ** 2 + 4 * (1 - z) * p_raw / vig
        return (math.sqrt(max(disc, 0)) - z) / (2 * (1 - z))

    p_a = _shin_one(p_raw_a)
    p_b = _shin_one(p_raw_b)
    total = p_a + p_b
    return p_a / total, p_b / total


def vig_pct(odds_a: float, odds_b: float) -> float:
    """Porcentaje de margen de la casa (vig) en una línea binaria."""
    return (1.0 / odds_a + 1.0 / odds_b - 1.0) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILIDADES POISSON
# ─────────────────────────────────────────────────────────────────────────────

def poisson_over_prob(lam_total: float, line: float) -> float:
    """
    P(X > line) bajo distribución Poisson con media lam_total.

    Las casas suelen usar líneas .5 (half-ball), por lo que
    P(over L.5) = P(X ≥ L+1) = 1 - Poisson_CDF(L, lam).

    Si la línea es entera (p.ej. 35.0), aplicamos el mismo CDF
    truncando al entero inferior.

    Args:
        lam_total: predicción del modelo (λ_home + λ_away, opcionalmente calibrada)
        line: línea del bookmaker (e.g. 33.5)

    Returns:
        Probabilidad de que el total supere la línea.
    """
    if lam_total <= 0:
        return 0.0
    floor_line = int(line)  # para líneas .5: floor(33.5) = 33
    return float(1.0 - poisson.cdf(floor_line, lam_total))


def poisson_under_prob(lam_total: float, line: float) -> float:
    """P(X ≤ line) bajo distribución Poisson. Complementario de over."""
    return 1.0 - poisson_over_prob(lam_total, line)


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILIDADES NEGATIVE BINOMIAL (tail O/U con dispersión tuneable)
# ─────────────────────────────────────────────────────────────────────────────

# α por defecto — ajustado offline vía
# `scripts/investigation/tune_alpha_via_cv.py` (method=grid_mle) sobre la val
# season 2025/2026 el 2026-04-22. Re-tunear cuando el modelo se reentrene
# (cambia μ ⇒ cambia el α óptimo). Valor en [0.001, 2.0]; si α cae fuera del
# rango sanity, el tuner escribe `alpha_in_range: false` y NO se promueve.
DEFAULT_NEGBIN_ALPHA: float = 0.007


def _nbinom_params(mu, alpha: float) -> tuple:
    """(μ, α) → (n, p) scipy. Vectorizado sobre μ.

    Parametrización scipy: X ~ NB(n, p) cuenta nro de fallos antes del n-ésimo éxito.
        mean = n(1-p)/p       var = n(1-p)/p²
    Con n = 1/α y p = 1/(1+α·μ):
        mean = (1/α)·(αμ) = μ                   ✓
        var  = mean/p = μ·(1+αμ) = μ + α·μ²     ✓
    """
    if alpha <= 0:
        raise ValueError(f"alpha debe ser > 0, recibido {alpha}")
    mu_arr = np.asarray(mu, dtype=float)
    if np.any(mu_arr <= 0):
        raise ValueError("mu debe ser > 0 para NegBin")
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu_arr)
    return n, p


def nbinom_over_prob(mu, line: float, alpha: float = DEFAULT_NEGBIN_ALPHA):
    """P(X > line) bajo X ~ NegBin(μ, α). Vectorizado sobre μ.

    Semántica de línea idéntica a `poisson_over_prob`: para líneas .5 (half-ball)
    `P(over L.5) = P(X ≥ L+1) = 1 - CDF(L, n, p)`. Para líneas enteras (push
    posible) truncamos al entero inferior, replicando el path Poisson actual —
    el manejo del push lo resuelve la casa ex-post, no el probability math.

    Args:
        mu: media predicha por el modelo (scalar o array-like).
        line: línea del bookmaker (e.g. 44.5).
        alpha: dispersión NegBin (default: DEFAULT_NEGBIN_ALPHA).

    Returns:
        P(over) — scalar si `mu` es scalar, ndarray si es array.
    """
    mu_arr = np.asarray(mu, dtype=float)
    scalar_input = mu_arr.ndim == 0
    mu_flat = np.atleast_1d(mu_arr)
    # Máscara µ<=0: no apostable, prob=0 (mantener contrato con poisson_over_prob).
    out = np.zeros(mu_flat.shape, dtype=float)
    valid = mu_flat > 0
    if np.any(valid):
        n, p = _nbinom_params(mu_flat[valid], float(alpha))
        floor_line = int(line)
        out[valid] = 1.0 - nbinom.cdf(floor_line, n, p)
    if scalar_input:
        return float(out[0])
    return out


def nbinom_under_prob(mu, line: float, alpha: float = DEFAULT_NEGBIN_ALPHA):
    """P(X ≤ line) bajo X ~ NegBin(μ, α). Complementario exacto de `nbinom_over_prob`.

    En líneas .5 `P(over) + P(under) == 1` exactamente (no hay push). En líneas
    enteras el floor también implica que `P(over) + P(under) == 1` porque
    ambos usan `floor(line)` como corte — el push queda enmascarado y lo
    liquida la casa. Matchea el path Poisson existente.
    """
    return 1.0 - nbinom_over_prob(mu, line, alpha)


def expected_value(p_model: float, odds: float) -> float:
    """
    EV de una apuesta: EV = p_model * odds - 1.
    Positivo → value bet. Negativo → la casa tiene ventaja.
    """
    return p_model * odds - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TEAM BIAS CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def load_team_bias(
    path: str | Path | None = None,
    model_trained_at: str | None = None,
) -> dict[str, dict[str, float]]:
    """
    Carga tabla de bias por (team_id, is_home) desde el JSON de calibración.

    Shape esperado del JSON: `corrections[team_id_str][is_home_str] = {n, raw_bias, shrunk_bias}`.
    Retorna `{str(team_id): {"0": shrunk_bias, "1": shrunk_bias}}` — solo el shrunk_bias,
    claves como strings para acceso directo por team_id/is_home serializados.

    Args:
        path: ruta al JSON. Default: data/model/team_bias_calibration_v2.json
        model_trained_at: si se pasa, compara con `model_trained_at` del JSON
            y emite warn si difieren (staleness check).

    Returns:
        dict anidado. Vacío si el archivo falta (+ warn-once por path).

    Raises:
        ValueError: si algún shrunk_bias es NaN/inf.
    """
    p = Path(path or _DEFAULT_BIAS_PATH)
    if not p.exists():
        key = str(p)
        if key not in _MISSING_BIAS_FILE_WARNED:
            log.warning("team_bias_calibration JSON no encontrado en %s — bias=0 para todos", key)
            _MISSING_BIAS_FILE_WARNED.add(key)
        return {}

    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if model_trained_at is not None:
        json_trained = payload.get("model_trained_at")
        if json_trained and json_trained != model_trained_at:
            log.warning(
                "team_bias staleness: JSON model_trained_at=%s, modelo actual=%s",
                json_trained, model_trained_at,
            )

    corrections = payload.get("corrections", {})
    table: dict[str, dict[str, float]] = {}
    for team_id_str, by_home in corrections.items():
        entry: dict[str, float] = {}
        for is_home_str, stats in by_home.items():
            shrunk = float(stats["shrunk_bias"])
            if not math.isfinite(shrunk):
                raise ValueError(
                    f"shrunk_bias no finito en team_id={team_id_str} is_home={is_home_str}: {shrunk}"
                )
            entry[str(is_home_str)] = shrunk
        table[str(team_id_str)] = entry

    log.info("team_bias cargado: %d equipos desde %s", len(table), p)
    return table


def shrink_bias(
    raw_bias: float,
    n: int,
    k: float = 5.0,
    prior_mu: float = 0.0,
) -> float:
    """
    Aplica shrinkage bayesiano (modelo normal-normal) a un bias crudo por equipo.

    Modelo:
      - Prior sobre el bias verdadero por equipo: bias ~ N(prior_mu, sigma2_prior).
      - Residuales por match: y - lambda ~ N(bias, sigma2_res).
      - Observamos n residuales; el bias crudo es su media.

    Posterior media (fórmula cerrada):
      shrunk = (n·raw / sigma2_res + prior_mu / sigma2_prior) /
               (n / sigma2_res + 1 / sigma2_prior)
             = raw · n / (n + k)   cuando prior_mu = 0 y k = sigma2_res / sigma2_prior.

    El JSON frozen `team_bias_calibration_v2.json` se generó con k=5.0 y prior_mu=0
    (verificado por reverse-engineering sobre las 40 filas n·raw/(n+k)≈shrunk, std=0.002).
    Mantener esa pareja garantiza que la regeneración reproduce la tabla actual
    cuando el modelo no cambia (verificación vía smoke `check_calibration_regen`).

    Args:
        raw_bias: media de residuales (y_true - lambda_pred) para (team_id, is_home).
        n: número de observaciones (matches).
        k: ratio sigma2_res / sigma2_prior. Default 5.0 — prior frozen.
        prior_mu: media del prior. Default 0.0 — "sin bias a priori".

    Returns:
        shrunk_bias posterior (float).
    """
    if n <= 0:
        return float(prior_mu)
    # Forma equivalente con prior_mu != 0:
    # w = n / (n + k); shrunk = w * raw + (1 - w) * prior_mu
    w = n / (n + k)
    return float(w * raw_bias + (1.0 - w) * prior_mu)


def apply_team_bias(
    lam,
    team_id,
    is_home,
    bias_table: dict[str, dict[str, float]],
    clip_min: float = 0.0,
):
    """
    Aplica el bias aditivamente a `lam`: `lam_cal = max(clip_min, lam + bias)`.

    Path primario vectorizado (np.ndarray / pd.Series). Scalar funciona vía broadcast.
    Equipo ausente en la tabla → bias=0 + warn-once por (team_id, is_home).

    Args:
        lam: λ raw predicha (scalar o array-like).
        team_id: id del equipo (scalar o array-like, alineado con lam).
        is_home: 0/1 (scalar o array-like).
        bias_table: output de `load_team_bias()`.
        clip_min: floor para el λ calibrado (Poisson requiere λ>0; default 0.0).

    Returns:
        λ calibrado con el mismo shape que el input.
    """
    lam_arr = np.asarray(lam, dtype=float)
    team_arr = np.asarray(team_id)
    home_arr = np.asarray(is_home)

    lam_flat = np.atleast_1d(lam_arr)
    team_flat = np.atleast_1d(team_arr)
    home_flat = np.atleast_1d(home_arr)

    n = lam_flat.shape[0]
    if team_flat.shape[0] == 1 and n > 1:
        team_flat = np.broadcast_to(team_flat, (n,))
    if home_flat.shape[0] == 1 and n > 1:
        home_flat = np.broadcast_to(home_flat, (n,))

    bias_vec = np.zeros(n, dtype=float)
    for i in range(n):
        tid = str(int(team_flat[i]))
        hid = str(int(home_flat[i]))
        row = bias_table.get(tid)
        if row is None or hid not in row:
            warn_key = (tid, hid)
            if warn_key not in _MISSING_TEAM_WARNED:
                log.warning("team_bias missing team_id=%s is_home=%s — bias=0", tid, hid)
                _MISSING_TEAM_WARNED.add(warn_key)
            continue
        bias_vec[i] = row[hid]

    out = np.maximum(lam_flat + bias_vec, clip_min)

    if lam_arr.ndim == 0:
        return float(out[0])
    return out.reshape(lam_arr.shape)


# ─────────────────────────────────────────────────────────────────────────────
# TEAM-WITH-MORE PRICING
# ─────────────────────────────────────────────────────────────────────────────

def p_home_more(
    lam_h,
    lam_w,
    method: str = "skellam",
    n_sim: int = 10_000,
    seed: int = 42,
):
    """
    Probabilidades del mercado binario `team_with_more`: P(H > A), P(H = A), P(A > H).

    El resultado son **3 probabilidades** porque Codere lista `draw` como selección
    explícita (no hay push-refund implícito — cada lado se cotiza por separado).

    Args:
        lam_h: λ home (scalar o 1D array).
        lam_w: λ away (scalar o 1D array, misma longitud que lam_h).
        method: "skellam" (closed-form Poisson, default) | "mc" (Monte Carlo).
        n_sim: simulaciones MC (ignorado si method="skellam").
        seed: seed del RNG para MC (determinismo).

    Returns:
        (p_home_strict, p_tie, p_away_strict) — cada uno es scalar o array según inputs.
        Suman 1.0 exactamente en Skellam y ~1.0 en MC.
    """
    lam_h_arr = np.asarray(lam_h, dtype=float)
    lam_w_arr = np.asarray(lam_w, dtype=float)
    scalar_input = lam_h_arr.ndim == 0 and lam_w_arr.ndim == 0

    lam_h_flat = np.atleast_1d(lam_h_arr)
    lam_w_flat = np.atleast_1d(lam_w_arr)

    if lam_h_flat.shape != lam_w_flat.shape:
        lam_h_flat, lam_w_flat = np.broadcast_arrays(lam_h_flat, lam_w_flat)

    if method == "skellam":
        p_tie = skellam.pmf(0, lam_h_flat, lam_w_flat)
        p_home = 1.0 - skellam.cdf(0, lam_h_flat, lam_w_flat)
        p_away = skellam.cdf(-1, lam_h_flat, lam_w_flat)
    elif method == "mc":
        rng = np.random.default_rng(seed)
        p_home = np.empty(lam_h_flat.shape, dtype=float)
        p_tie = np.empty(lam_h_flat.shape, dtype=float)
        p_away = np.empty(lam_h_flat.shape, dtype=float)
        for i in range(lam_h_flat.size):
            sh = rng.poisson(lam_h_flat.flat[i], size=n_sim)
            sw = rng.poisson(lam_w_flat.flat[i], size=n_sim)
            p_home.flat[i] = float(np.mean(sh > sw))
            p_tie.flat[i] = float(np.mean(sh == sw))
            p_away.flat[i] = float(np.mean(sh < sw))
    else:
        raise ValueError(f"method debe ser 'skellam' o 'mc', recibí: {method!r}")

    if scalar_input:
        return float(p_home[0]), float(p_tie[0]), float(p_away[0])
    return p_home, p_tie, p_away
