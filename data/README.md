# Fonctionnement de la Machine Learning Supervisée

Projet de Fin d'Études (PFE) — Cycle Licence en Éducation (CLE), Spécialité Mathématiques
**École Normale Supérieure de Tétouan** — Université Abdelmalek Essaadi
Année universitaire 2024-2025

> Réalisé par : Wejdan Ghailane, Doha Et-touimy, Nada Maghouz
> Encadré par : Pr. El hichami Outman

## Description

Ce projet est une étude complète de l'apprentissage automatique supervisé,
centrée sur deux grandes familles d'algorithmes :

- **Régression** : régression linéaire (simple, multiple, polynomiale),
  descente de gradient (GD) et descente de gradient stochastique (SGD)
- **Classification** : Machines à Vecteurs de Support (SVM)

Le projet combine les fondements mathématiques (méthode des moindres carrés,
fonctions de coût, hyperplan optimal, marge maximale...) avec une mise en
œuvre pratique en Python, appliquée à deux cas concrets :

1. **Prédiction du prix de vente de logements** (dataset Ames Housing) —
   comparaison entre méthode analytique, GD et SGD.
2. **Classification d'emails (Spam / Non Spam)** avec un SVM à noyau linéaire.

## Structure du dépôt

```
.
├── src/
│   ├── regression_ames_housing.py    # Régression sur le dataset Ames Housing
│   ├── case_study_prix_logement.py   # Étude de cas (5 logements) : analytique / GD / SGD
│   └── svm_classification_spam.py    # Classification SVM (détection de spam)
├── data/                              # Jeux de données (à ajouter, voir ci-dessous)
├── docs/
│   └── rapport_pfe.pdf                # Rapport complet du projet (84 pages)
├── requirements.txt
└── README.md
```

## Technologies utilisées

- **Python 3**
- **NumPy** — calcul numérique et opérations matricielles
- **Pandas** — manipulation et nettoyage des données
- **Matplotlib** — visualisation des résultats
- **Scikit-learn** — implémentation du modèle SVM

## Installation

```bash
git clone <url-de-votre-depot>
cd pfe-ml-project
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Jeux de données

Les scripts s'appuient sur deux datasets qui ne sont pas inclus dans le dépôt
(pour garder celui-ci léger) :

| Fichier              | Utilisé par                       | Source |
|----------------------|------------------------------------|--------|
| `data/AmesHousing.csv` | `regression_ames_housing.py`     | [Kaggle - Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) |
| `data/mails.csv`       | `svm_classification_spam.py`     | Dataset simplifié (fautes d'orthographe / mots-clés suspects → Spam), à ajouter manuellement |

Placez ces fichiers dans le dossier `data/` avant d'exécuter les scripts
correspondants.

## Utilisation

```bash
python src/regression_ames_housing.py
python src/case_study_prix_logement.py     # ne nécessite aucun fichier externe
python src/svm_classification_spam.py
```

## Résumé du rapport

Ce travail explore l'apprentissage supervisé à travers la régression linéaire
et la classification SVM, en développant leurs fondements mathématiques puis
en illustrant leur implémentation en Python. Le rapport complet (théorie,
démonstrations mathématiques, technologies, applications) est disponible
dans [`docs/rapport_pfe.pdf`](docs/rapport_pfe.pdf).

## Pistes d'amélioration

- Explorer d'autres algorithmes de régression (Ridge, Lasso)
- Explorer d'autres algorithmes de classification (arbres de décision, réseaux de neurones)
- Appliquer les modèles à des jeux de données plus complexes
- Tester des techniques d'optimisation plus avancées (Adam, RMSprop)

## Licence

Projet académique — usage éducatif.
