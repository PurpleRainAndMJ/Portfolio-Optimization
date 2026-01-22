import ccxt
import pandas as pd
import time

class BinanceLoader:
    def __init__(self):
        self.exchange = ccxt.binance()

    def fetch_crypto_data(self, symbols, timeframe='1d', limit=500):
        """
        Récupère les prix de clôture pour une liste de symboles.
        Ex de symboles : ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        """
        data = {}
        for symbol in symbols:
            print(f"Récupération de {symbol}...")
            # On récupère les bougies (OHLCV)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            data[symbol] = df['close']
            time.sleep(0.1)  # Respecter les limites de l'API
        
        return pd.DataFrame(data)
