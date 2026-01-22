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
        # Rendement historique moyen
        self.mu = expected_returns.mean_historical_return(prices)
        # Matrice de covariance Ledoit-Wolf (plus stable pour les cryptos)
        self.S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        
        # Benchmark par défaut si non fourni
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
        
        total_cap = sum(market_caps.values())
        mkt_weights = {tik: cap / total_cap for tik, cap in market_caps.items()}
        return mkt_weights, mkt_prior

    def get_black_litterman(self, market_caps, investor_views, view_confidences=None):
        # 1. Nettoyage et vérification des données
        if not isinstance(self.S, pd.DataFrame):
            self.S = pd.DataFrame(self.S, index=self.prices.columns, columns=self.prices.columns)
        
        delta = black_litterman.market_implied_risk_aversion(self.benchmark_prices)
        market_prior = black_litterman.market_implied_prior_returns(market_caps, delta, self.S)
        
        # 2. Cas sans vues : retour à l'équilibre
        if not investor_views:
            ef = EfficientFrontier(market_prior, self.S)
            weights = ef.max_sharpe()
            return ef.clean_weights(), ef.portfolio_performance(verbose=False)

        # 3. Préparation des vues (Series)
        views_series = pd.Series(investor_views)
        
        # 4. Configuration Omega (Incertitude)
        if view_confidences and len(view_confidences) == len(investor_views):
            omega_method = "idzorek"
            conf = view_confidences
        else:
            omega_method = None
            conf = None

        bl = BlackLittermanModel(
            cov_matrix=self.S,
            pi=market_prior,
            absolute_views=views_series,
            omega=omega_method,
            view_confidences=conf,
            tau=0.05
        )
        
        self.ret_bl = bl.bl_returns()
        self.S_bl = bl.bl_cov()
        
        # 5. Optimisation finale
        ef = EfficientFrontier(self.ret_bl, self.S_bl)
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)

    def generate_random_portfolios(self, n_samples=2000):
        """Simulation de Monte Carlo pour la Frontière Efficiente."""
        results = []
        for _ in range(n_samples):
            w = np.random.random(len(self.prices.columns))
            w /= np.sum(w)
            ret = np.dot(w, self.mu)
            vol = np.sqrt(np.dot(w.T, np.dot(self.S, w)))
            results.append((vol, ret))
        return np.array(results)