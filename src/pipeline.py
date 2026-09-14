"""
pipeline.py
-----------
Point d'entrée unique : nettoie les données brutes, entraîne et évalue le
modèle, sauvegarde toutes les sorties (data/processed, outputs/figures,
outputs/metrics).

Usage :
    python src/pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.data_cleaning import load_and_clean
from src.modeling import train_and_evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raw_path = PROJECT_ROOT / "data" / "raw" / "mini_etudiant_13_panne_equipement.csv"
    processed_path = PROJECT_ROOT / "data" / "processed" / "panne_equipement_clean.csv"
    outputs_dir = PROJECT_ROOT / "outputs"

    print(f"[1/2] Nettoyage : {raw_path.name}")
    df_clean, clean_report = load_and_clean(raw_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    print(json.dumps(clean_report, indent=2, ensure_ascii=False, default=str))

    print("\n[2/2] Modélisation")
    results = train_and_evaluate(df_clean, outputs_dir)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nTerminé. Données propres -> {processed_path}")
    print(f"Figures & métriques -> {outputs_dir}")


if __name__ == "__main__":
    main()
