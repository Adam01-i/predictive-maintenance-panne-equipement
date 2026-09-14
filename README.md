# Maintenance prédictive — Prédiction de panne d'équipement à 30 jours

Projet de data science : nettoyage d'un jeu de données capteurs industriels
(température, vibration, usure, etc.) et entraînement d'un modèle de
classification pour prédire si un équipement va tomber en panne dans les
30 prochains jours.

## Pourquoi ce projet

Ce dépôt part d'un exercice pédagogique de nettoyage de données (notebook
d'origine : chargement + nettoyage d'un CSV mal formaté). Il a été
entièrement restructuré et enrichi pour en faire une démonstration complète
de bout en bout : diagnostic de qualité de données → nettoyage corrigé →
exploration → entraînement de modèle → évaluation → interprétation.

## Résultats obtenus

Sur un jeu de test de 180 lignes tenu à l'écart de l'entraînement :

| Métrique | Valeur |
|---|---|
| Modèle retenu | Random Forest (`n_estimators=300`, `max_depth=6`, `class_weight="balanced"`) |
| ROC-AUC (test) | **0.786** |
| Accuracy | 0.806 |
| Précision / Rappel classe "panne" | 0.48 / 0.38 |

Voir `outputs/metrics/metrics.json` pour le détail complet et
`outputs/figures/` pour les visualisations (matrice de confusion, courbe
ROC, importance des variables, distribution de la cible).

Les variables les plus prédictives sont **la vibration**, **le nombre de
jours depuis la dernière maintenance** et **la température** — cohérent
avec l'intuition métier (usure mécanique + carence d'entretien).

> Le rappel modéré sur la classe minoritaire ("panne") est attendu : le
> jeu de données est petit (900 lignes), synthétique et volontairement
> bruité pour un exercice pédagogique. Les résultats sont illustratifs,
> pas un modèle prêt pour la production.

## Structure du dépôt

```
.
├── data/
│   ├── raw/                 # CSV brut original (non modifié)
│   └── processed/           # CSV nettoyé, généré par le pipeline
├── notebooks/
│   ├── 01_exploration_nettoyage.ipynb   # diagnostic + nettoyage + EDA
│   └── 02_modelisation.ipynb            # entraînement + évaluation modèle
├── src/
│   ├── data_cleaning.py     # fonctions de nettoyage réutilisables/testées
│   ├── modeling.py          # préprocessing, entraînement, évaluation, figures
│   └── pipeline.py          # script unique bout-en-bout (CLI)
├── outputs/
│   ├── figures/             # PNG générés (matrice confusion, ROC, etc.)
│   └── metrics/             # metrics.json (résultats chiffrés)
├── tests/
│   └── test_data_cleaning.py
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Exécution

Pipeline complet (nettoyage + modélisation + figures) en une commande :

```bash
python -m src.pipeline
```

Ou pas à pas via les notebooks, dans l'ordre :
1. `notebooks/01_exploration_nettoyage.ipynb`
2. `notebooks/02_modelisation.ipynb`

Tests :

```bash
python -m pytest tests/ -v
# ou, sans pytest :
python tests/test_data_cleaning.py
```

## Données

Le jeu de données (`data/raw/mini_etudiant_13_panne_equipement.csv`, 900
lignes, 13 colonnes) contient des mesures capteurs simulées pour trois
types d'équipements (pompe, convoyeur, compresseur) répartis sur 3 sites
(A, B, C), avec une cible binaire `panne_30j`.

Il faisait partie d'un lot de 20 jeux de données d'exercice fournis par un
cours (`datasets.zip`) ; seul celui-ci est utilisé par ce projet — les 19
autres (churn, défaut de crédit, retard de livraison, diagnostic) relèvent
d'exercices distincts et n'ont pas été inclus dans ce dépôt.

## Nettoyage : bugs corrigés par rapport à la version d'origine

1. **Perte silencieuse de données liée aux décimales françaises.** La
   version d'origine convertissait les colonnes en numérique
   (`pd.to_numeric`) *avant* de remplacer la virgule décimale par un point.
   Résultat : des valeurs comme `"23,00"` devenaient `NaN` au lieu d'être
   lues comme `23.0`, puis étaient silencieusement remplacées par la
   médiane — une vraie donnée était perdue sans avertissement. **Corrigé** :
   la conversion virgule → point est faite avant toute coercition numérique.
2. **`drop_duplicates()` appelé sans réassignation** (`df.drop_duplicates()`
   au lieu de `df = df.drop_duplicates()`) : n'avait aucun effet. Corrigé.
3. **Médiane d'imputation calculée trop tôt**, sur des colonnes pas encore
   nettoyées (donc biaisée par les valeurs mal parsées). Corrigé : le calcul
   se fait après nettoyage complet.
4. **Logique dupliquée 3 à 4 fois** (nettoyage colonnes texte, imputation,
   suppression de doublons, réapparaissant à plusieurs endroits du
   notebook) → simplifiée en un pipeline linéaire unique et testé
   (`src/data_cleaning.py`).
5. **Chemin de fichier codé en dur** → paramétré en argument de fonction.
6. **Valeurs aberrantes jamais traitées** (seulement comptées). Décision
   prise : les signaler dans une colonne `outlier_flag` plutôt que les
   supprimer ou les imputer aveuglément, car elles peuvent être un signal
   réel de dysfonctionnement pertinent pour la prédiction de panne.
7. **`on_bad_lines='skip'`** (masque silencieusement les lignes corrompues)
   remplacé par `'warn'` pour visibilité.

## Modélisation : choix méthodologiques

- **`random_state=42`** fixé partout (split, forêt aléatoire) — absent de
  la version d'origine, qui de toute façon n'entraînait aucun modèle.
- **Split train/test stratifié** (80/20) pour préserver le ratio de classes
  (~19% de pannes) — évite un biais d'évaluation.
- **Prévention de fuite de données (data leakage)** : le préprocesseur
  (standardisation + one-hot encoding) est ajusté uniquement sur le train,
  jamais sur l'ensemble des données avant le split.
- **`class_weight="balanced"`** sur les deux modèles candidats plutôt qu'un
  ré-échantillonnage naïf, pour gérer le déséquilibre de classes.
- **Sélection de modèle par validation croisée** (5 folds, ROC-AUC) entre
  régression logistique et forêt aléatoire, puis évaluation finale sur le
  jeu de test tenu à l'écart.
- **Métriques adaptées au déséquilibre** (ROC-AUC, précision/rappel/F1 par
  classe) plutôt que la seule accuracy, trompeuse ici (~81% en prédisant
  toujours "pas de panne").

## Limites connues / suite possible

- Jeu de données petit et synthétique : résultats illustratifs pour un
  portfolio, pas un modèle de production.
- Pas de recherche d'hyperparamètres poussée (`GridSearchCV`) pour garder
  un temps d'exécution raisonnable.
- Amélioration possible avec des données de séries temporelles par
  équipement plutôt qu'un instantané.

## Stack technique

Python 3.11, pandas, numpy, scikit-learn, matplotlib, seaborn.

## Licence

MIT — voir [LICENSE](LICENSE).
