"""Tests de non-régression pour src/data_cleaning.py.

Lancer avec : python -m pytest tests/ -v
(ou, sans pytest installé : python tests/test_data_cleaning.py)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_cleaning import clean, NUMERIC_COLS, TEXT_COLS, TARGET_COL


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature_c": ["55.45", "39,31", "NA", ""],
            "vibration": [7.02, 1.11, 2.0, None],
            "humidite_pct": [28.63, 74.84, 52.79, 60.0],
            "pression_bar": [2.93, 3.38, 7.57, 4.0],
            "heures_utilisation_jour": [19.39, 12.35, 5.74, 10.0],
            "nb_alertes_7j": [3, "5", "23,00", 1],
            "derniere_maintenance_jours": [132, 251, 294, 100],
            "nb_interventions_12m": [15, 5, 11, 2],
            "consommation_kwh_j": [370.3, 582.8, 242.0, 300.0],
            "indice_usure": [8.66, 41.36, 9.41, 20.0],
            "site": ["B", " C ", "", "A"],
            "equipement_type": ["pompe", "pompe", "convoyeur", "pompe"],
            "panne_30j": [0, 0, 0, 1],
        }
    )


def test_decimal_comma_is_recovered_not_dropped():
    """Bug historique : '23,00' devenait NaN puis médiane. Doit maintenant
    être correctement interprété comme 23.0."""
    df_clean, _ = clean(_toy_df())
    assert df_clean.loc[2, "nb_alertes_7j"] == 23.0


def test_no_missing_values_after_cleaning():
    df_clean, _ = clean(_toy_df())
    assert df_clean[NUMERIC_COLS].isna().sum().sum() == 0
    assert df_clean[TEXT_COLS].isna().sum().sum() == 0


def test_no_duplicate_rows_after_cleaning():
    df = pd.concat([_toy_df(), _toy_df().iloc[[0]]], ignore_index=True)
    df_clean, report = clean(df)
    assert report["doublons_supprimes"] >= 1
    assert df_clean.duplicated().sum() == 0


def test_target_is_integer_binary():
    df_clean, _ = clean(_toy_df())
    assert set(df_clean[TARGET_COL].unique()) <= {0, 1}
    assert df_clean[TARGET_COL].dtype.kind in "iu"


def test_outlier_flag_column_added():
    df_clean, _ = clean(_toy_df())
    assert "outlier_flag" in df_clean.columns


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"OK   - {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL - {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    sys.exit(1 if failed else 0)
