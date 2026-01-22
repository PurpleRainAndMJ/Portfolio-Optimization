import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, black_litterman
from pypfopt.black_litterman import BlackLittermanModel

class PortfolioEngine:
    def __init__(self, prices, benchmark_prices=None):
        """
        prices: DataFrame des actifs du portefeuille.
        benchmark_prices: Series d'un indice de référence (ex: BTC) pour calculer l'aversion au risque du marché.
        """
        self.prices = prices
        # On utilise le rendement historique moyen comme "Prior" de base si BL n'est pas utilisé
        self.mu = expected_returns.mean_historical_return(prices)
        # Matrice de covariance (Ledoit-Wolf est souvent plus robuste pour la crypto)
        self.S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        
        # Si aucun benchmark n'est fourni, on prend une moyenne simple des actifs
        self.benchmark_prices = benchmark_prices if benchmark_prices is not None else prices.mean(axis=1)

    # --- Méthodes Classiques (Markowitz) ---
    def get_max_sharpe(self):
        ef = EfficientFrontier(self.mu, self.S)
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)

    # --- Méthodes Black-Litterman ---
    def get_market_equilibrium(self, market_caps):
        """Calcule les poids 'neutres' du marché selon la capitalisation."""
        delta = black_litterman.market_implied_risk_aversion(self.benchmark_prices)
        mkt_prior = black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
        # Un simple calcul des poids par capitalisation relative
        total_cap = sum(market_caps.values())
        mkt_weights = {tik: cap / total_cap for tik, cap in market_caps.items()}
        return mkt_weights, mkt_prior

    def get_black_litterman(self, market_caps, investor_views, view_confidences=None):
        # 1. S'assurer que la matrice de covariance est un DataFrame avec les bons noms d'actifs
        if not isinstance(self.S, pd.DataFrame):
            self.S = pd.DataFrame(self.S, index=self.prices.columns, columns=self.prices.columns)

        # 2. Calculer l'aversion au risque et le Prior du marché
        delta = black_litterman.market_implied_risk_aversion(self.benchmark_prices)
        market_prior = black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
        
        # 3. Transformer les vues en Pandas Series pour un mapping parfait des actifs
        # Cela évite les erreurs de correspondance entre le dictionnaire et la matrice S
        views_series = pd.Series(investor_views)

        # 4. Initialisation du modèle avec le nom complet "idzorek"
        # On passe explicitement les paramètres pour éviter les erreurs de type en Python 3.14
        try:
            bl = BlackLittermanModel(
                cov_matrix=self.S,
                pi=market_prior,
                absolute_views=views_series,
                omega="idzorek",  # Utilisation du nom complet au lieu de l'alias "idz"
                view_confidences=view_confidences,
                tau=0.05
            )
            
            self.ret_bl = bl.bl_returns()
            self.S_bl = bl.bl_cov()
            
            ef = EfficientFrontier(self.ret_bl, self.S_bl)
            weights = ef.max_sharpe()
            return ef.clean_weights(), ef.portfolio_performance(verbose=False)
            
        except TypeError as e:
            print(f"⚠️ Erreur de type détectée : {e}")
            print("Tentative de repli sur un calcul manuel d'Omega...")
            # Fallback : Si l'alias "idzorek" échoue, on laisse le modèle calculer Omega par défaut (None)
            bl = BlackLittermanModel(self.S, pi=market_prior, absolute_views=views_series)
            self.ret_bl = bl.bl_returns()
            self.S_bl = bl.bl_cov()
            ef = EfficientFrontier(self.ret_bl, self.S_bl)
            return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    
    def generate_random_portfolios(self, n_samples=2000):
        """Génère des portefeuilles aléatoires pour tracer le nuage de points."""
        results = []
        for _ in range(n_samples):
            w = np.random.random(len(self.prices.columns))
            w /= np.sum(w)
            ret = np.dot(w, self.mu) # Ou ret_bl si calculé
            vol = np.sqrt(np.dot(w.T, np.dot(self.S, w)))
            results.append((vol, ret))
        return np.array(results)