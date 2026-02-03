# 🟣 Crypto Portfolio Engine: Markowitz & Black-Litterman (Interactive Edition)

Ce projet est une plateforme d'**ingénierie financière quantitative** permettant de construire, d'optimiser et de visualiser des portefeuilles de cryptomonnaies. Il combine la rigueur mathématique des modèles de **Markowitz** et de **Black-Litterman** avec une interface utilisateur moderne et interactive.


---

## 🚀 Fonctionnalités Clés

* **Interface Streamlit Interactive** : Sélectionnez vos actifs, ajustez vos convictions (Views) et lancez l'optimisation en un clic.
* **Modèle Black-Litterman Robuste** : 
    * Intégration de la méthode d'**Idzorek** pour lier mathématiquement la confiance de l'investisseur à l'incertitude du modèle.
    * Calcul dynamique de l'aversion au risque ($\delta$) et des rendements d'équilibre.
* **Optimisation de Markowitz (Moyenne-Variance)** : Calcul de la frontière efficiente via simulations de Monte Carlo.
* **Pipeline de Données Réelles** : Connecteur multi-actifs via l'API **Binance** (CCXT).
* **Visualisations Avancées** :
    * Comparaison des poids (Market vs Markowitz vs Black-Litterman).
    * Nuage de points de la Frontière Efficiente avec identification du ratio de Sharpe optimal.

---

## 📊 Fondamentaux Mathématiques

### 1. Théorie Moderne du Portefeuille (MPT)
L'objectif est de minimiser la variance $\sigma_p^2$ pour un niveau de rendement attendu :
$$\min_{w} w^T \Sigma w$$
Sous contrainte de plein investissement $\sum w_i = 1$.

### 2. Modèle de Black-Litterman
Le moteur utilise une approche bayésienne pour fusionner les rendements d'équilibre du marché ($\Pi$) avec les vues subjectives de l'investisseur ($Q$) :
$$E[R] = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1} [(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]$$
L'incertitude des vues ($\Omega$) est calibrée via la méthode d'Idzorek pour garantir la cohérence des allocations.

---

## 📂 Architecture du Projet

```text
Portfolio-Optimization/
├── src/
│   ├── data_loader.py    # Extraction et nettoyage des données Binance
│   ├── optimizer.py      # Moteur mathématique (BL & Markowitz)
│   └── visualizer.py     # Fonctions de rendu graphique
├── output/               # Exports PNG des résultats
├── app.py                # Interface Web Streamlit
├── main.py               # Script d'exécution en ligne de commande
├── tests_quant.py        # Suite de tests unitaires
└── requirements.txt      # Dépendances (PyPortfolioOpt, Streamlit, etc.)
