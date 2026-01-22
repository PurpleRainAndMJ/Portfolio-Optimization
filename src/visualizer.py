import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Configuration du style
sns.set_theme(style="whitegrid")

def _save_or_show(filename):
    """Gère la sauvegarde pour Docker/Local et évite les erreurs d'affichage."""
    if not os.path.exists('output'):
        os.makedirs('output')
    
    path = os.path.join('output', filename)
    plt.savefig(path)
    
    # Si on est dans Streamlit, on ne fait pas plt.show() pour éviter les conflits
    # Streamlit utilise son propre st.pyplot(fig)
    if "STREAMLIT_RUN" not in os.environ:
        try:
            plt.show()
        except Exception:
            pass

def plot_strategy_comparison(strategies_weights, title="Comparaison des Allocations"):
    df = pd.DataFrame(strategies_weights)
    fig, ax = plt.subplots(figsize=(12, 7))
    df.plot(kind='bar', ax=ax, width=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Allocation (%)")
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        if p.get_height() > 0.01:
            ax.annotate(f"{p.get_height():.1%}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontsize=9)
    plt.tight_layout()
    _save_or_show("comparaison_strategies.png")
    return fig

def plot_efficient_frontier_cloud(random_portfolios, optimized_point, title="Frontière Efficiente"):
    """
    Trace le nuage de points des portefeuilles simulés.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calcul du ratio de Sharpe pour la couleur
    sharpe_ratios = random_portfolios[:, 1] / random_portfolios[:, 0]
    
    scatter = ax.scatter(random_portfolios[:, 0], random_portfolios[:, 1], 
                         c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.5)
    
    # Ajout du point optimal (l'étoile rouge)
    ax.scatter(optimized_point[1], optimized_point[0], 
                color='red', marker='*', s=200, label="Portefeuille Optimal (BL)")
    
    plt.colorbar(scatter, label='Ratio de Sharpe')
    plt.title(title, fontsize=14)
    plt.xlabel('Volatilité (Risque)')
    plt.ylabel('Rendement Attendu')
    plt.legend()
    
    _save_or_show("frontiere_efficiente.png")
    return fig

def plot_performance(returns_dict, title="Performance Cumulée"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, returns in returns_dict.items():
        cum_returns = (1 + returns).cumprod()
        ax.plot(cum_returns, label=f"{name}")
    plt.title(title)
    plt.legend()
    plt.ylabel("Multiplicateur de Richesse")
    plt.grid(True, alpha=0.3)
    _save_or_show("performance_cumulee.png")
    return fig