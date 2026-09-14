# Maintenance prédictive

## Prédire une panne d'équipement dans les 30 jours

Projet de data science de bout en bout : partir de mesures de capteurs
industriels imparfaites, construire un jeu de données exploitable, puis
entraîner un modèle de classification capable d'estimer le risque de panne.

Le dépôt met l'accent sur la **qualité des données**, la **reproductibilité**
et une évaluation adaptée à une cible déséquilibrée. Il s'agit d'un projet
pédagogique et portfolio, pas d'un système de décision industrielle.

## Résultats

Le pipeline sélectionne le modèle sur une validation croisée stratifiée à
5 folds, puis mesure sa performance sur un jeu de test séparé (180 lignes).
Les résultats ci-dessous correspondent aux artefacts actuellement générés
dans [`outputs/metrics/metrics.json`](outputs/metrics/metrics.json).

| Indicateur | Résultat |
|---|---:|
| Modèle retenu | Régression logistique pondérée |
| ROC-AUC moyen en validation croisée | 0,762 ± 0,037 |
| ROC-AUC sur le test | **0,775** |
| Accuracy sur le test | 0,728 |
| Précision classe `panne` | 0,377 |
| Rappel classe `panne` | **0,676** |
| F1-score classe `panne` | 0,484 |

Le rappel de 0,676 signifie que le modèle détecte environ deux pannes sur
trois dans cet échantillon de test. En contrepartie, la précision reste
limitée : plusieurs alertes sont des faux positifs. C'est un compromis
attendu pour un problème où manquer une panne peut être plus coûteux que
déclencher une inspection supplémentaire.

Visualisations produites :

- [Courbe ROC](outputs/figures/roc_curve.png)
- [Matrice de confusion](outputs/figures/confusion_matrix.png)
- [Importance des variables](outputs/figures/feature_importance.png)
- [Répartition de la cible](outputs/figures/target_distribution.png)

## Pipeline de traitement

```text
CSV brut
  -> normalisation des colonnes et des formats
  -> conversion des décimales françaises
  -> imputation des valeurs manquantes
  -> détection des outliers via la règle IQR
  -> split train/test stratifié
  -> standardisation + encodage one-hot dans un Pipeline sklearn
  -> comparaison régression logistique / random forest
  -> évaluation et génération des artefacts
```

Les valeurs aberrantes ne sont pas supprimées automatiquement : elles sont
conservées et signalées dans la colonne `outlier_flag`, car une mesure
extrême peut justement indiquer une dégradation de l'équipement.

## Jeu de données

Le fichier [`data/raw/mini_etudiant_13_panne_equipement.csv`](data/raw/mini_etudiant_13_panne_equipement.csv)
contient 900 observations de capteurs simulés, réparties sur trois sites et
trois types d'équipements : pompe, convoyeur et compresseur.

Variables utilisées :

- Capteurs : température, vibration, humidité, pression et consommation.
- Contexte d'utilisation : heures d'utilisation, alertes récentes,
  maintenance, interventions et indice d'usure.
- Catégories : `site` et `equipement_type`.
- Cible : `panne_30j`, indicateur binaire d'une panne dans les 30 jours.

Le fichier nettoyé est écrit dans
[`data/processed/panne_equipement_clean.csv`](data/processed/panne_equipement_clean.csv).
Il contient 900 lignes et 14 colonnes, dont `outlier_flag`.

## Installation

Prérequis : Python 3.11 ou version compatible avec les dépendances du
projet.

```bash
git clone https://github.com/Adam01-i/predictive-maintenance-panne-equipement.git
cd predictive-maintenance-panne-equipement

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Sous Windows, l'activation de l'environnement virtuel est :

```powershell
.venv\Scripts\Activate.ps1
```

## Utilisation

Lancer le pipeline complet depuis la racine du projet :

```bash
python -m src.pipeline
```

Cette commande effectue successivement le nettoyage, l'entraînement,
l'évaluation et la génération des figures et métriques.

Pour explorer le projet étape par étape, ouvrir les notebooks dans cet ordre :

1. [`notebooks/01_exploration_nettoyage.ipynb`](notebooks/01_exploration_nettoyage.ipynb)
2. [`notebooks/02_modelisation.ipynb`](notebooks/02_modelisation.ipynb)

## Tests

Les tests vérifient notamment la conversion des décimales françaises, la
gestion des valeurs manquantes, la suppression des doublons, le type de la
cible et la création du signal d'outlier.

```bash
python -m pytest tests/ -v
```

Une exécution sans pytest est également possible :

```bash
python tests/test_data_cleaning.py
```

## Structure

```text
.
├── data/
│   ├── raw/                 # données originales
│   └── processed/           # données nettoyées générées
├── notebooks/               # exploration et modélisation guidées
├── outputs/
│   ├── figures/             # visualisations générées
│   └── metrics/             # résultats JSON
├── src/
│   ├── data_cleaning.py     # chargement, nettoyage et rapport qualité
│   ├── modeling.py          # entraînement, évaluation et figures
│   └── pipeline.py          # point d'entrée bout en bout
├── tests/                   # tests de non-régression
├── requirements.txt
└── LICENSE
```

## Choix techniques

- Conversion de la virgule décimale avant toute coercition numérique, afin
  de ne pas transformer une mesure valide en valeur manquante.
- Imputation des colonnes numériques par la médiane et des colonnes
  catégorielles par le mode, après nettoyage des formats.
- Suppression des lignes dont la cible est invalide, sans imputation de la
  variable à prédire.
- Préprocesseur ajusté uniquement sur le train pour éviter la fuite de
  données : `StandardScaler` pour les variables numériques et
  `OneHotEncoder` pour les catégories.
- `class_weight="balanced"` pour tenir compte du déséquilibre entre pannes
  et non-pannes.
- Comparaison d'une régression logistique et d'une random forest selon la
  ROC-AUC, puis évaluation finale sur un test jamais utilisé pour la
  sélection.
- `random_state=42` fixé pour rendre l'expérience reproductible.

## Limites et pistes d'amélioration

Le jeu de données est synthétique, de petite taille et représente un instantané
plutôt qu'une vraie série temporelle par équipement. Les métriques ne doivent
donc pas être interprétées comme une garantie de performance en production.

Pour aller plus loin :

- collecter des historiques horodatés par équipement ;
- ajuster le seuil de décision selon le coût métier des faux négatifs ;
- réaliser une recherche d'hyperparamètres et une validation temporelle ;
- ajouter un suivi de dérive des données et une comparaison avec une règle
  métier simple.

## Licence

Projet distribué sous licence MIT. Voir [`LICENSE`](LICENSE).
