from src.data_loader import BinanceLoader
from src.optimizer import PortfolioEngine
from src.visualizer import plot_strategy_comparison, plot_performance
import pandas as pd

# --- 1. CONFIGURATION & DONNÉES FONDAMENTALES ---
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT']
BENCHMARK_SYMBOL = 'BTC/USDT' # Le marché directeur

# Capitalisations boursières APPROXIMATIVES (en Milliards $ pour l'exemple)
# NOTE: En prod, ces données devraient être récupérées dynamiquement via une API comme CoinGecko.
MARKET_CAPS = {
    'BTC/USDT': 850.0,
    'ETH/USDT': 280.0,
    'BNB/USDT': 45.0,
    'SOL/USDT': 40.0,
    'LINK/USDT': 8.0
}

# --- 2. DÉFINITION DES VUES DE L'INVESTISSEUR (C'est VOUS) ---
# "Je pense que SOL va surperformer énormément et que ETH sera légèrement positif"
INVESTOR_VIEWS = {
    'SOL/USDT': 0.35,  # Vue très optimiste : +35% attendu
    'ETH/USDT': 0.10,  # Vue modérément optimiste : +10% attendu
    # On n'a pas d'avis sur BTC, BNB ou LINK, donc on laisse le modèle suivre le marché.
}

# Confiance dans nos vues (entre 0 et 1)
VIEW_CONFIDENCES = [
    0.8, # Très confiant sur SOL
    0.5  # Moyennement confiant sur ETH
]

# --- 3. ACQUISITION DES DONNÉES ---
print("📡 Récupération des données de marché...")
loader = BinanceLoader()
# On charge les actifs + le benchmark
all_prices = loader.fetch_crypto_data(SYMBOLS, limit=500)

# On sépare les prix des actifs du prix du benchmark
asset_prices = all_prices[SYMBOLS]
# Si BTC est dans la liste, on l'utilise comme benchmark
benchmark_prices = all_prices[BENCHMARK_SYMBOL] if BENCHMARK_SYMBOL in all_prices else asset_prices.mean(axis=1)

# --- 4. INITIALISATION DU MOTEUR ---
engine = PortfolioEngine(asset_prices, benchmark_prices)

# --- 5. EXÉCUTION DES 3 STRATÉGIES ---

print("\n--- Calcul Stratégie 1 : Équilibre de Marché (Neutre) ---")
# Ce que le marché détient "en moyenne" selon la capitalisation
weights_mkt, prior_mkt = engine.get_market_equilibrium(MARKET_CAPS)

print("\n--- Calcul Stratégie 2 : Markowitz Historique (Pure Data) ---")
# Se base uniquement sur le passé, ignore les caps et les vues
weights_mkz, perf_mkz = engine.get_max_sharpe()

print("\n--- Calcul Stratégie 3 : Black-Litterman (Hybride) ---")
# Combine l'équilibre de marché ET vos vues subjectives
weights_bl, perf_bl = engine.get_black_litterman(MARKET_CAPS, INVESTOR_VIEWS, VIEW_CONFIDENCES)

# --- 6. RÉSULTATS & VISUALISATION ---

print("\n" + "="*40)
print(" RÉSULTATS BLACK-LITTERMAN")
print("="*40)
print(f"Rendement Attendu BL : {perf_bl[0]:.2%}")
print(f"Volatilité BL        : {perf_bl[1]:.2%}")
print(f"Sharpe Ratio BL      : {perf_bl[2]:.2f}")
print("-" * 40)
print("Vues injectées :")
for ticker, view in INVESTOR_VIEWS.items():
    print(f"  - {ticker} : {view:+.0%}")
print("="*40)

# Préparation des données pour le graphique comparatif
strategies_to_plot = {
    "1. Marché (Caps)": weights_mkt,
    "2. Markowitz (Passé)": weights_mkz,
    "3. Black-Litterman (Vues)": weights_bl
}

# Lancement du graphique principal
print("📊 Génération du graphique comparatif...")
plot_strategy_comparison(strategies_to_plot, title="Impact des Vues de l'Investisseur sur l'Allocation (Black-Litterman)")