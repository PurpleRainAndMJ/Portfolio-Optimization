import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuration du style
sns.set_theme(style="whitegrid")

def plot_strategy_comparison(strategies_weights, title="Comparaison des Allocations Stratégiques"):
    """
    Compare visuellement les poids des différentes stratégies cote à cote.
    strategies_weights: dictionnaire de dictionnaires. Ex:
    {
        "Marché (Neutre)": {'BTC': 0.6, ...},
        "Markowitz (Historique)": {'BTC': 0.2, ...},
        "Black-Litterman (Vues)": {'BTC': 0.4, ...}
    }
    """
    # Conversion en DataFrame pour faciliter le plotting groupé
    df = pd.DataFrame(strategies_weights)
    
    # Création du graphique
    ax = df.plot(kind='bar', figsize=(12, 7), width=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel("Allocation (%)")
    plt.xlabel("Actifs")
    plt.legend(title="Stratégies")
    plt.xticks(rotation=45)
    
    # Ajout des pourcentages au-dessus des barres
    for p in ax.patches:
        if p.get_height() > 0.01: # On n'affiche pas les 0%
            ax.annotate(f"{p.get_height():.1%}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontsize=9)
            
    plt.tight_layout()
    plt.show()

# (Garder les anciennes fonctions plot_performance et plot_weights si désiré)
def plot_performance(returns_dict, title="Performance Cumulée Comparée"):
    plt.figure(figsize=(12, 6))
    for name, returns in returns_dict.items():
        cum_returns = (1 + returns).cumprod()
        plt.plot(cum_returns, label=f"{name}")
    plt.title(title)
    plt.legend()
    plt.ylabel("Multiplicateur de Richesse (Base 1.0)")
    plt.grid(True, alpha=0.3)
    plt.show()