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
  build_name_map()                    → dict
  normalize_team(name)                → str
"""
from __future__ import annotations

import math
from typing import Optional

from scipy.stats import poisson

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


def expected_value(p_model: float, odds: float) -> float:
    """
    EV de una apuesta: EV = p_model * odds - 1.
    Positivo → value bet. Negativo → la casa tiene ventaja.
    """
    return p_model * odds - 1.0
