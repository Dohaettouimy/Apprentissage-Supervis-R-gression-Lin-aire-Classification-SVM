"""
Étude de cas : Prédiction du prix des logements par régression linéaire
=========================================================================

Petit jeu de données (5 logements) : surface (m²) -> prix (kDH).
Comparaison de 3 méthodes : moindres carrés (analytique), descente de
gradient (GD) et descente de gradient stochastique (SGD).
"""

import random
import numpy as np
import matplotlib.pyplot as plt

X = np.array([50, 60, 80, 100, 120])
y = np.array([850, 900, 1100, 1500, 1700])
n = len(X)


# ---------------------------------------------------------------------------
# 1. Méthode analytique : moindres carrés
# ---------------------------------------------------------------------------
def methode_analytique(x, y):
    n = len(x)
    a = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - (np.sum(x)) ** 2)
    b = (np.sum(y) - a * np.sum(x)) / n
    cost = (1 / (2 * n)) * np.sum((a * x + b - y) ** 2)
    return a, b, cost


# ---------------------------------------------------------------------------
# 2. Descente de gradient (batch)
# ---------------------------------------------------------------------------
def descente_gradient(x, y, learning_rate=0.00001, epochs=150):
    n = len(x)
    m, b = 0.0, 0.0
    cost_history = []

    for epoch in range(epochs):
        y_pred = m * x + b
        error = y_pred - y

        gradient_m = (2 / n) * np.dot(error, x)
        gradient_b = (2 / n) * np.sum(error)

        m -= learning_rate * gradient_m
        b -= learning_rate * gradient_b

        mse = np.mean(error ** 2)
        cost_history.append(mse)

    return m, b, cost_history


# ---------------------------------------------------------------------------
# 3. Descente de gradient stochastique (SGD)
# ---------------------------------------------------------------------------
def descente_gradient_stochastique(x, y, learning_rate=0.00001, epochs=150, seed=42):
    random.seed(seed)
    n = len(x)
    m, b = 0.0, 0.0
    cost_history = []
    indices = list(range(n))

    for epoch in range(epochs):
        random.shuffle(indices)
        for i in indices:
            x_i, y_i = x[i], y[i]
            y_pred = m * x_i + b
            error = y_pred - y_i

            gradient_m = 2 * error * x_i
            gradient_b = 2 * error

            m -= learning_rate * gradient_m
            b -= learning_rate * gradient_b

        y_all_pred = m * x + b
        epoch_cost = np.mean((y_all_pred - y) ** 2)
        cost_history.append(epoch_cost)

    return m, b, cost_history


if __name__ == "__main__":
    # --- Méthode analytique ---
    a, b, cost = methode_analytique(X, y)
    print(f"[Analytique] Coefficient directeur a = {a:.2f}")
    print(f"[Analytique] Ordonnée à l'origine b = {b:.2f}")
    print(f"[Analytique] Fonction de coût J = {cost:.2f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="blue", label="Données observées")
    plt.plot(X, a * X + b, color="red", label=f"Droite ajustée: y = {a:.2f}x + {b:.2f}")
    plt.title("Régression linéaire - Méthode des moindres carrés")
    plt.xlabel("Surface (m²)")
    plt.ylabel("Prix (kDH)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Descente de gradient ---
    m_gd, b_gd, cost_gd = descente_gradient(X, y)
    print(f"\n--- Résultats avec Descente de Gradient (GD) ---")
    print(f"Paramètre 'm' final : {m_gd:.4f}")
    print(f"Paramètre 'b' final : {b_gd:.4f}")
    print(f"Modèle final : Prix = {m_gd:.2f} * Surface + {b_gd:.2f}")

    surface_test = 75
    prix_estime = m_gd * surface_test + b_gd
    print(f"Prédiction pour une surface de {surface_test} m² : {prix_estime:.2f} kDH")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(X, y, color="blue", label="Données réelles")
    ax1.plot(X, m_gd * X + b_gd, color="green", label="Régression (GD)")
    ax1.set_title("Régression Linéaire (GD)")
    ax1.set_xlabel("Surface (m²)")
    ax1.set_ylabel("Prix (kDH)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(len(cost_gd)), cost_gd, color="green")
    ax2.set_title("Évolution de l'Erreur (MSE) - GD")
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Erreur Quadratique Moyenne")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # --- SGD ---
    m_sgd, b_sgd, cost_sgd = descente_gradient_stochastique(X, y)
    print(f"\n=== Résultats du modèle SGD ===")
    print(f"Pente 'm' finale : {m_sgd:.4f}")
    print(f"Ordonnée à l'origine 'b' finale : {b_sgd:.4f}")
    print(f"Modèle final : Prix = {m_sgd:.2f} * Surface + {b_sgd:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.scatter(X, y, color="black", zorder=5, label="Données réelles")
    ax1.plot(X, m_sgd * X + b_sgd, color="red", linestyle="--", linewidth=2, label="Régression SGD")
    ax1.set_title("Modèle de Régression Final (SGD)", fontsize=14)
    ax1.set_xlabel("Surface (m²)")
    ax1.set_ylabel("Prix (kDH)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(len(cost_sgd)), cost_sgd, color="red", alpha=0.9, label="Erreur SGD")
    ax2.set_title("Évolution de l'Erreur (SGD)", fontsize=14)
    ax2.set_xlabel("Époques")
    ax2.set_ylabel("Erreur Quadratique Moyenne (MSE)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, which="both")
    plt.tight_layout()
    plt.show()
