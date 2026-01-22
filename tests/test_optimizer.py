import pytest
import pandas as pd
import numpy as np
from src.optimizer import PortfolioEngine

@pytest.fixture
def dummy_data():
    """Génère des données fictives pour les tests."""
    data = pd.DataFrame({
        'BTC': [100, 105, 102, 110],
        'ETH': [100, 101, 104, 103]
    })
    return data

def test_weights_sum_to_one(dummy_data):
    """Vérifie que la somme des poids est toujours égale à 1 (100%)."""
    engine = PortfolioEngine(dummy_data)
    weights, _ = engine.get_max_sharpe()
    assert pytest.approx(sum(weights.values())) == 1.0

def test_negative_prices_error():
    """Le moteur doit lever une erreur si les prix sont négatifs (impossible)."""
    bad_data = pd.DataFrame({'BTC': [-10, 20, 30]})
    with pytest.raises(Exception):
        PortfolioEngine(bad_data)