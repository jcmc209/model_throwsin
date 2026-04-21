"""
Normalizador de nombres de equipo — single source para scripts/evaluation
=========================================================================
Clave canónica determinista y pura para cruzar predicciones
(`dataset.parquet` / `predictions_*.parquet`) con cuotas
(`odds_codere.parquet`, `odds_22bet.parquet`).

Consumido por `evaluate_vs_market.py` y `value_bets.py`. No duplicar la
lógica de normalización en ningún otro lugar — importar desde acá.
"""
from __future__ import annotations

import re
import unicodedata

_CLUB_SUFFIX_RE = re.compile(r"\s+(UD|FC|CF|CD|SD|RCD|RC)\b", flags=re.IGNORECASE)
_TEAM_ALIASES: dict[str, str] = {
    "atletico madrid": "atletico",
    "deportivo alaves": "alaves",
}


def _normalize_team_name(name: str) -> str:
    """
    Clave canónica para cruces entre predicciones (`dataset.parquet`) y cuotas
    (`odds_codere.parquet`). Determinista y pura: NFKD → quita diacríticos →
    lowercase → strip de sufijos de club → alias fijos.
    """
    if not isinstance(name, str):
        return ""
    stripped = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    stripped = stripped.strip().lower()
    stripped = _CLUB_SUFFIX_RE.sub("", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return _TEAM_ALIASES.get(stripped, stripped)
