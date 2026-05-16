"""
Ingest Codere Odds CSV → Parquet (Unified Schema)
==================================================
Convierte `data/reference/odds_codere.csv` a `data/reference/odds_codere.parquet`
con el mismo esquema que `odds_22bet.parquet`, de modo que ambas fuentes puedan
mergearse trivialmente para análisis cross-bookmaker.

Mapeos principales:
  mercado="Total de Saques de Banda Más/Menos"
     selection="Más de X"   → market_type="total_over_under", side="over",  line=X
     selection="Menos de X" → market_type="total_over_under", side="under", line=X

  mercado="Equipo con Más Saques de Banda"
     selection=home_team    → market_type="team_with_more",  side="home", line=None
     selection=away_team    → market_type="team_with_more",  side="away", line=None
     selection="Empate"     → market_type="team_with_more",  side="draw", line=None

Esquema resultante (alineado con odds_22bet.parquet):
  home_team          str
  away_team          str
  scraped_at         datetime64[ns, UTC]
  bookmaker          "codere"
  market_type        "total_over_under" | "team_with_more"
  line               float | NaN
  side               "over" | "under" | "home" | "away" | "draw"
  odds               float
  raw_market_name    str (trazabilidad)
  raw_selection      str (trazabilidad)

Uso:
  python scripts/odds/ingest_codere_csv.py
  python scripts/odds/ingest_codere_csv.py --input custom.csv --output custom.parquet
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/reference/odds_codere.csv")
PARQUET_PATH = Path("data/reference/odds_codere.parquet")

OVER_UNDER_RE = re.compile(r"^(M[aá]s|Menos) de (\d+(?:\.\d+)?)$", re.IGNORECASE)


def _classify_selection(selection: str, home_team: str, away_team: str) -> tuple[str, str | None, float | None]:
    """
    Devuelve (market_type, side, line).
    - total_over_under → side=over|under, line=float
    - team_with_more   → side=home|away|draw, line=None
    """
    sel = str(selection).strip()

    m = OVER_UNDER_RE.match(sel)
    if m:
        direction = m.group(1).lower()
        line = float(m.group(2))
        side = "over" if direction.startswith("m") and "á" in direction or direction == "mas" or direction == "más" else "under"
        # Robustez: "Más" empieza por M y tiene >3 chars, "Menos" empieza por M y tiene 5 chars
        side = "over" if direction.lower().startswith("m") and not direction.lower().startswith("men") else "under"
        return "total_over_under", side, line

    # Team with more — comparación tolerante (Codere usa "Atlético", "Rayo", etc.)
    sel_norm = sel.lower().strip()
    if sel_norm == "empate":
        return "team_with_more", "draw", None
    if sel_norm in home_team.lower() or home_team.lower() in sel_norm:
        return "team_with_more", "home", None
    if sel_norm in away_team.lower() or away_team.lower() in sel_norm:
        return "team_with_more", "away", None

    # Fallback: devolvemos team_with_more con side=unknown para no perder la fila
    return "team_with_more", "unknown", None


def ingest(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    n_in = len(df)

    df["scraped_at"] = pd.to_datetime(df["scraped_at"], utc=True)

    out_rows = []
    for _, r in df.iterrows():
        market_type, side, line = _classify_selection(
            r["selection"], r["home_team"], r["away_team"]
        )
        out_rows.append({
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "scraped_at": r["scraped_at"],
            "bookmaker": "codere",
            "market_type": market_type,
            "line": line,
            "side": side,
            "odds": float(r["cuota"]),
            "raw_market_name": r["mercado"],
            "raw_selection": r["selection"],
        })

    out = pd.DataFrame(out_rows)
    out["line"] = out["line"].astype("float64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Input:  {input_path}  ({n_in} filas)")
    print(f"Output: {output_path} ({len(out)} filas)")
    print(f"\nDistribución market_type x side:")
    print(out.groupby(["market_type", "side"]).size().to_string())

    # Sanity
    unknown = (out["side"] == "unknown").sum()
    if unknown:
        print(f"\nATENCIÓN: {unknown} filas con side='unknown' (selection no mapeado)")
        print(out[out["side"] == "unknown"][["home_team", "away_team", "raw_selection"]].to_string(index=False))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Codere CSV odds → unified parquet")
    parser.add_argument("--input", default=str(CSV_PATH))
    parser.add_argument("--output", default=str(PARQUET_PATH))
    args = parser.parse_args()

    ingest(Path(args.input), Path(args.output))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
