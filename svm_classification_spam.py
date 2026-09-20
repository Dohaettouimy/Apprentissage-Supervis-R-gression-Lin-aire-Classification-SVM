"""
Classification d'Emails avec Support Vector Machine (SVM)
============================================================

Objectif : classifier automatiquement des emails en "Spam" (1) ou
"Non Spam" (0) à partir de deux caractéristiques :
    - nombre de fautes d'orthographe
    - nombre de mots-clés suspects

Dataset attendu : data/mails.csv (colonnes séparées par ';')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def load_data(path="data/mails.csv"):
    df = pd.read_csv(path, delimiter=";")
    print(df.head())
    return df


def prepare_features(df):
    """Sépare les caractéristiques (X) de la cible (Y)."""
    X1 = df.iloc[:, 0].values  # nombre de fautes d'orthographe
    X2 = df.iloc[:, 1].values  # nombre de mots-clés suspects
    Y = df.iloc[:, -1].values  # Spam (1) / Non Spam (0)

    X1 = X1.reshape(len(X1), 1)
    X2 = X2.reshape(len(X2), 1)
    X = np.hstack((X1, X2))

    return X, Y


def train_svm(X, Y, kernel="linear"):
    model = SVC(kernel=kernel)
    model.fit(X, Y)
    return model


def plot_decision_boundary(model, X, Y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="coolwarm", edgecolors="k")

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors="k", levels=[0], linestyles=["-"])
    plt.xlabel("Nombre de fautes d'orthographe")
    plt.ylabel("Nombre de mots-clés suspects")
    plt.title("Frontière de décision SVM (kernel linéaire)")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    X, Y = prepare_features(df)

    model = train_svm(X, Y, kernel="linear")
    print(f"\nScore (précision) sur les données d'entraînement : {model.score(X, Y):.2f}")

    plot_decision_boundary(model, X, Y)

    # Exemples de prédiction
    exemple_1 = [[6, 6]]  # 6 fautes, 6 mots-clés
    exemple_2 = [[1, 1]]  # 1 faute, 1 mot-clé

    print(f"Email avec 6 fautes et 6 mots-clés -> "
          f"{'Spam' if model.predict(exemple_1)[0] == 1 else 'Non Spam'}")
    print(f"Email avec 1 faute et 1 mot-clé -> "
          f"{'Spam' if model.predict(exemple_2)[0] == 1 else 'Non Spam'}")
