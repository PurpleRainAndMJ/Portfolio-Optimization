import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import BinanceLoader
from src.optimizer import PortfolioEngine
from src.visualizer import plot_strategy_comparison, plot_efficient_frontier_cloud

st.set_page_config(page_title="Crypto Quant Optimizer", layout="wide")

st.title("🟣 Crypto Portfolio Optimizer")
st.markdown("Optimisation avancée via **Markowitz** et **Black-Litterman**.")

# --- SIDEBAR : CONFIGURATION ---
st.sidebar.header("⚙️ Paramètres")

available_tickers = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'ADA/USDT', 'AVAX/USDT']

selected_symbols = st.sidebar.multiselect(
    "Actifs à inclure",
    available_tickers,
    default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    key="selector_symbols"
)

# --- SIDEBAR : VUES (Convictions) ---
st.sidebar.subheader("🎯 Vos convictions (Views)")
views = {}
confidences = []

for symbol in selected_symbols:
    # Ajout d'une clé unique basée sur le symbole pour chaque checkbox, slider et input
    use_view = st.sidebar.checkbox(f"Avoir une vue sur {symbol}", value=False, key=f"cb_{symbol}")
    if use_view:
        view_val = st.sidebar.number_input(f"Rendement attendu {symbol} (%)", -100, 200, 10, key=f"val_{symbol}") / 100
        conf = st.sidebar.slider(f"Confiance {symbol}", 0.0, 1.0, 0.5, key=f"conf_{symbol}")
        views[symbol] = view_val
        confidences.append(conf)

# --- BOUTON DE LANCEMENT ---
# Ajout d'une clé unique 'btn_opt' pour éviter l'erreur DuplicateElementId
if st.sidebar.button("Lancer l'Optimisation", key="btn_opt"):
    if len(selected_symbols) < 2:
        st.error("⚠️ Veuillez sélectionner au moins 2 actifs pour l'optimisation.")
    else:
        with st.spinner('🚀 Récupération des données et calculs en cours...'):
            try:
                # 1. Chargement des données
                loader = BinanceLoader()
                prices = loader.fetch_crypto_data(selected_symbols, limit=500)
                
                # 2. Initialisation du moteur
                # On utilise BTC comme benchmark s'il est présent, sinon la moyenne
                bench = prices['BTC/USDT'] if 'BTC/USDT' in prices.columns else None
                engine = PortfolioEngine(prices, benchmark_prices=bench)
                
                # 3. Calculs des stratégies
                mkt_caps = {s: 100 for s in selected_symbols} 
                w_mkz, perf_mkz = engine.get_max_sharpe()
                
                # Vérification des confidences pour Black-Litterman
                current_conf = confidences if (len(confidences) == len(views) and len(views) > 0) else None
                w_bl, perf_bl = engine.get_black_litterman(mkt_caps, views, current_conf)
                
                # --- AFFICHAGE DES RÉSULTATS ---
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Répartition des Poids")
                    strategies = {
                        "Markowitz (Hist)": w_mkz,
                        "Black-Litterman": w_bl
                    }
                    fig_comp = plot_strategy_comparison(strategies)
                    st.pyplot(fig_comp)

                with col2:
                    st.subheader("🎯 Frontière Efficiente")
                    random_ports = engine.generate_random_portfolios(n_samples=1500)
                    # Utilisation des perfs de BL pour le point optimal
                    fig_ef = plot_efficient_frontier_cloud(random_ports, (perf_bl[0], perf_bl[1]))
                    st.pyplot(fig_ef)

                # Table de performance finale
                st.write("### 📈 Métriques de Performance (Black-Litterman)")
                metrics_df = pd.DataFrame({
                    "Métrique": ["Rendement Annuel", "Volatilité Annuelle", "Ratio de Sharpe"],
                    "Valeur": [f"{perf_bl[0]:.2%}", f"{perf_bl[1]:.2%}", f"{perf_bl[2]:.2f}"]
                })
                st.table(metrics_df)

            except Exception as e:
                st.error(f"❌ Une erreur est survenue : {e}")