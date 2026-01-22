# 🟣 Portfolio Risk & Optimization (Crypto Edition)

Ce projet est un outil d'**ingénierie financière quantitative** permettant de construire, d'optimiser et de stress-tester un portefeuille d'actifs numériques (Cryptomonnaies). Il utilise l'API **Binance** pour les données réelles et applique les modèles de **Markowitz** pour l'arbitrage rendement/risque.

---

## 🚀 Fonctionnalités Clés

* **Extraction Automatisée** : Récupération des données OHLCV via l'API Binance (`ccxt`).
* **Optimisation Markowitz** : Calcul de la Frontière Efficiente pour maximiser le **Ratio de Sharpe**.
* **Gestion du Risque** :
    * Calcul de la **Value-at-Risk (VaR)** historique à 95%.
    * Analyse du **Maximum Drawdown** (perte maximale historique).
    * Optimisation sous contrainte de **Volatilité Cible**.
* **Visualisation Avancée** : Comparaison des performances cumulées entre le portefeuille optimisé et un portefeuille naïf (1/N).

---

## 📂 Structure du Dépôt

```text
portfolio-optimization/
├── src/
│   ├── data_loader.py    # Client API Binance & nettoyage
│   ├── optimizer.py      # Moteur de calcul (Markowitz & Risk metrics)
│   └── visualizer.py     # Graphiques Matplotlib/Seaborn
├── main.py               # Script d'exécution principal
├── requirements.txt      # Dépendances du projet
└── README.md             # Documentation