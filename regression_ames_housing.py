"""
Régression linéaire simple - Prédiction du prix de vente (Ames Housing Dataset)
=================================================================================

Ce module compare 3 approches pour estimer les coefficients b0 (intercept)
et b1 (pente) du modèle y = b0 + b1*x :

    1. Méthode analytique (moindres carrés)
    2. Descente de gradient (batch)
    3. Descente de gradient stochastique (SGD)

Variable explicative (x) : Gr Liv Area (surface habitable)
Variable cible (y)       : SalePrice (prix de vente)

Dataset attendu : data/AmesHousing.csv
(télécharger depuis Kaggle : https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path="data/AmesHousing.csv"):
    """Charge et prépare les données (surface habitable / prix de vente)."""
    data = pd.read_csv(path)
    data = data[["Gr Liv Area", "SalePrice"]].dropna()
    x = data["Gr Liv Area"].values
    y = data["SalePrice"].values
    return x, y


def normalize(x):
    """Normalisation standard : x_norm = (x - moyenne) / écart-type."""
    return (x - np.mean(x)) / np.std(x)


# ---------------------------------------------------------------------------
# 1. Méthode analytique (moindres carrés)
# ---------------------------------------------------------------------------
def regression_analytique(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerateur = np.sum((x - x_mean) * (y - y_mean))
    denominateur = np.sum((x - x_mean) ** 2)
    b1 = numerateur / denominateur
    b0 = y_mean - b1 * x_mean
    return b0, b1


# ---------------------------------------------------------------------------
# 2. Descente de gradient (batch)
# ---------------------------------------------------------------------------
def descente_gradient(x, y, learning_rate=0.01, epochs=1000):
    b0, b1 = 0.0, 0.0
    n = len(x)
    cost_history = []

    for epoch in range(epochs):
        y_pred = b0 + b1 * x
        error = y_pred - y
        cost = (1 / n) * np.sum(error ** 2)
        cost_history.append(cost)

        grad_b0 = (2 / n) * np.sum(error)
        grad_b1 = (2 / n) * np.sum(error * x)

        b0 -= learning_rate * grad_b0
        b1 -= learning_rate * grad_b1

    return b0, b1, cost_history


# ---------------------------------------------------------------------------
# 3. Descente de gradient stochastique (SGD)
# ---------------------------------------------------------------------------
def descente_gradient_stochastique(x, y, learning_rate=0.01, epochs=100):
    b0, b1 = 0.0, 0.0
    n = len(x)
    cost_history = []

    for epoch in range(epochs):
        total_cost = 0
        for i in range(n):
            xi, yi = x[i], y[i]
            y_pred = b0 + b1 * xi
            error = y_pred - yi

            b0 -= learning_rate * error
            b1 -= learning_rate * error * xi
            total_cost += error ** 2

        cost_history.append(total_cost / n)

    return b0, b1, cost_history


def plot_regression(x, y, y_pred, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue", label="Données réelles", alpha=0.5)
    plt.plot(x, y_pred, color="red", label="Régression")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cost_history(cost_history, title):
    plt.plot(range(len(cost_history)), cost_history, color="green")
    plt.xlabel("Époques")
    plt.ylabel("Erreur quadratique moyenne (MSE)")
    plt.title(title)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    x_raw, y = load_data()
    x = normalize(x_raw)

    # Méthode analytique
    b0, b1 = regression_analytique(x, y)
    print(f"[Analytique] y = {b0:.2f} + {b1:.2f} * x")
    plot_regression(x, y, b0 + b1 * x, "Régression linéaire (analytique)",
                     "Surface habitable normalisée", "Prix de vente")

    # Descente de gradient (batch)
    b0_gd, b1_gd, cost_gd = descente_gradient(x, y, learning_rate=0.01, epochs=1000)
    print(f"[Gradient Descent] y = {b0_gd:.2f} + {b1_gd:.2f} * x_normalisé")
    plot_regression(x, y, b0_gd + b1_gd * x, "Régression linéaire (descente de gradient)",
                     "Surface habitable normalisée", "Prix de vente")
    plot_cost_history(cost_gd, "Évolution de la fonction de coût (GD)")

    # SGD
    b0_sgd, b1_sgd, cost_sgd = descente_gradient_stochastique(x, y, learning_rate=0.01, epochs=100)
    print(f"[SGD] y = {b0_sgd:.2f} + {b1_sgd:.2f} * x_normalisé")
    plot_regression(x, y, b0_sgd + b1_sgd * x, "Régression linéaire (SGD)",
                     "Surface habitable normalisée", "Prix de vente")
    plot_cost_history(cost_sgd, "Convergence de la descente de gradient stochastique")
