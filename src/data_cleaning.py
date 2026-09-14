from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NUMERIC_COLS = [
    "temperature_c",
    "vibration",
    "humidite_pct",
    "pression_bar",
    "heures_utilisation_jour",
    "nb_alertes_7j",
    "derniere_maintenance_jours",
    "nb_interventions_12m",
    "consommation_kwh_j",
    "indice_usure",
]
TEXT_COLS = ["site", "equipement_type"]
TARGET_COL = "panne_30j"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Charge le CSV brut. `on_bad_lines='warn'` pour ne pas masquer les
    lignes corrompues silencieusement (l'original utilisait 'skip')."""
    return pd.read_csv(path, on_bad_lines="warn")


def _fix_decimal_comma(series: pd.Series) -> pd.Series:
    """Remplace la virgule décimale française par un point AVANT toute
    conversion numérique. Doit être appelé pendant que la colonne est
    encore de type texte/objet."""
    return series.astype(str).str.replace(",", ".", regex=False)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie le DataFrame et retourne (df_propre, rapport).

    Le rapport contient les statistiques de nettoyage utiles pour le README
    (nombre de valeurs corrigées, doublons supprimés, etc.).
    """
    report: dict = {}
    df = df.copy()

    # 1. Normalisation des noms de colonnes et suppression des espaces
    df.columns = df.columns.str.strip()

    # 2. Nettoyage des colonnes texte
    for col in TEXT_COLS:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "NA": np.nan})

    # 3. Colonnes numériques : virgule -> point AVANT conversion, puis coercition
    for col in NUMERIC_COLS + [TARGET_COL]:
        if col not in df.columns:
            continue
        df[col] = _fix_decimal_comma(df[col])
        df[col] = df[col].replace({"NA": np.nan, "nan": np.nan})
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Doublons (bug corrigé : réassignation)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["doublons_supprimes"] = n_before - len(df)

    # 5. Valeurs manquantes -> imputation médiane / mode (après nettoyage complet)
    missing_report = {}
    for col in NUMERIC_COLS:
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            missing_report[col] = {"n_manquants": int(n_missing), "valeur_utilisee": float(median_val)}
    for col in TEXT_COLS:
        n_missing = df[col].isna().sum()
        if n_missing:
            mode_val = df[col].mode(dropna=True)
            mode_val = mode_val.iloc[0] if len(mode_val) else "inconnu"
            df[col] = df[col].fillna(mode_val)
            missing_report[col] = {"n_manquants": int(n_missing), "valeur_utilisee": mode_val}
    report["valeurs_manquantes"] = missing_report

    # 6. Cible : lignes sans cible valide sont supprimées (on ne peut pas
    #    imputer une variable à prédire sans introduire de fuite de données)
    n_before_target = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    report["lignes_supprimees_cible_invalide"] = n_before_target - len(df)

    # 7. Détection d'outliers (IQR) -> signalés, pas supprimés
    outlier_mask = pd.Series(False, index=df.index)
    outlier_detail = {}
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        col_outliers = (df[col] < lower) | (df[col] > upper)
        if col_outliers.any():
            outlier_detail[col] = int(col_outliers.sum())
        outlier_mask |= col_outliers
    df["outlier_flag"] = outlier_mask
    report["outliers_par_colonne"] = outlier_detail
    report["lignes_avec_outlier"] = int(outlier_mask.sum())

    report["n_lignes_finales"] = len(df)
    report["n_colonnes_finales"] = df.shape[1]

    return df, report


def load_and_clean(raw_path: str | Path) -> tuple[pd.DataFrame, dict]:
    df_raw = load_raw(raw_path)
    return clean(df_raw)


if __name__ == "__main__":
    import json

    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "mini_etudiant_13_panne_equipement.csv"
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "panne_equipement_clean.csv"

    df_clean, report = load_and_clean(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)

    logger.info("Nettoyage terminé -> %s", out_path)
    logger.info(json.dumps(report, indent=2, ensure_ascii=False, default=str))
