"""
modeling.py
-----------
Pipeline de modélisation pour prédire une panne d'équipement dans les 30
prochains jours (colonne cible `panne_30j`, classification binaire).

Choix méthodologiques (documentés pour transparence) :
- `random_state=42` fixé partout où une composante aléatoire existe
  (split, forêts aléatoires) pour la reproductibilité (bug absent dans le
  notebook original qui n'entraînait de toute façon aucun modèle).
- `train_test_split(..., stratify=y)` pour préserver le ratio de classes
  (~19% de pannes) dans train et test.
- Séparation stricte train/test AVANT tout ajustement de préprocesseur
  (`Pipeline` scikit-learn) pour éviter toute fuite de données (data leakage) :
  le `StandardScaler` et le `OneHotEncoder` sont ajustés uniquement sur le
  train, puis appliqués au test.
- `class_weight="balanced"` pour compenser le déséquilibre de classes plutôt
  que du sur-échantillonnage naïf.
- Métriques adaptées au déséquilibre : ROC-AUC, précision/rappel/F1 par
  classe, matrice de confusion. L'accuracy seule serait trompeuse (~81%
  en prédisant toujours "pas de panne").
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC_FEATURES = [
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
CATEGORICAL_FEATURES = ["site", "equipement_type"]
TARGET = "panne_30j"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def train_and_evaluate(df: pd.DataFrame, outputs_dir: Path) -> dict:
    """Entraîne deux modèles candidats, sélectionne le meilleur par
    validation croisée (ROC-AUC), évalue sur un jeu de test tenu à l'écart
    et sauvegarde les figures/métriques dans `outputs_dir`."""

    figures_dir = outputs_dir / "figures"
    metrics_dir = outputs_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=6,
        ),
    }

    cv_results = {}
    fitted_pipelines = {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        cv_results[name] = {"roc_auc_mean": float(scores.mean()), "roc_auc_std": float(scores.std())}
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc_mean"])
    best_pipe = fitted_pipelines[best_name]

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    test_auc = roc_auc_score(y_test, y_proba)

    # --- Figures ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {best_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title(f"Courbe ROC — {best_name} (AUC={test_auc:.3f})")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Importance des variables (si random forest sélectionné, sinon coefficients)
    feature_names = list(
        best_pipe.named_steps["preprocess"].get_feature_names_out()
    )
    model_step = best_pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    else:
        importances = np.abs(model_step.coef_[0])

    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    imp_series.tail(15).plot(kind="barh", ax=ax, color="#3b6ea5")
    ax.set_title(f"Importance des variables — {best_name}")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    # Distribution de la cible (utile pour montrer le déséquilibre)
    fig, ax = plt.subplots(figsize=(4, 4))
    df[TARGET].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#3b6ea5", "#d9534f"])
    ax.set_xticklabels(["Pas de panne (0)", "Panne (1)"], rotation=0)
    ax.set_title("Répartition de la cible panne_30j")
    fig.tight_layout()
    fig.savefig(figures_dir / "target_distribution.png", dpi=150)
    plt.close(fig)

    results = {
        "modele_selectionne": best_name,
        "cv_roc_auc_par_modele": cv_results,
        "test_roc_auc": float(test_auc),
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    clean_path = project_root / "data" / "processed" / "panne_equipement_clean.csv"
    df = pd.read_csv(clean_path)
    res = train_and_evaluate(df, project_root / "outputs")
    print(json.dumps(res, indent=2, ensure_ascii=False))
