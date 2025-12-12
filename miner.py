import ccxt
import time
import pandas as pd
import numpy as np
import os

# ================= CONFIGURAZIONE =================
SYMBOL = 'BTC/USDT'
EXCHANGE = ccxt.bingx()
DEPTH = 20
FILE_NAME = 'training_data.csv'
LOOK_AHEAD = 60            
TARGET_PROFIT = 0.0004     
TIMEFRAME = '1m'
# ==================================================

def calculate_indicators(closes):
    series = pd.Series(closes)
    # RSI (0-1 Normalized)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
    return current_rsi / 100.0, 0 # Ritorniamo solo RSI per semplicità qui, o aggiungi SMA se vuoi

def get_normalized_features():
    try:
        book = EXCHANGE.fetch_order_book(SYMBOL, limit=DEPTH)
        trades = EXCHANGE.fetch_trades(SYMBOL, limit=100)
        ticker = EXCHANGE.fetch_ticker(SYMBOL)
        ohlcv = EXCHANGE.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=30)
        
        # --- ORDER BOOK ---
        asks_p = np.array([x[0] for x in book['asks']]) 
        asks_v = np.array([x[1] for x in book['asks']]) 
        bids_p = np.array([x[0] for x in book['bids']])
        bids_v = np.array([x[1] for x in book['bids']])
        asks_usd = asks_p * asks_v
        bids_usd = bids_p * bids_v
        
        features = []
        
        def calc_obi(bid_val, ask_val):
            t = bid_val + ask_val
            return (bid_val - ask_val) / t if t > 0 else 0

        # Features 0-4: OBI (Range -1 a 1)
        features.append(calc_obi(bids_usd[0], asks_usd[0])) 
        features.append(calc_obi(bids_usd[1], asks_usd[1])) 
        features.append(calc_obi(bids_usd[2], asks_usd[2])) 
        features.append(calc_obi(np.sum(bids_usd[3:10]), np.sum(asks_usd[3:10]))) 
        features.append(calc_obi(np.sum(bids_usd[10:20]), np.sum(asks_usd[10:20]))) 
        
        # Features 5-6: Concentrazione (Range 0 a 1)
        features.append(bids_usd[0] / (np.sum(bids_usd) + 1)) 
        features.append(asks_usd[0] / (np.sum(asks_usd) + 1)) 
        
        # --- VOLUME & ICEBERG ---
        now = EXCHANGE.milliseconds()
        recent = [t for t in trades if t['timestamp'] > (now - 5000)]
        buy_usd = sum([t['cost'] if t.get('cost') else t['amount']*t['price'] for t in recent if t['side']=='buy'])
        sell_usd = sum([t['cost'] if t.get('cost') else t['amount']*t['price'] for t in recent if t['side']=='sell'])
        vol_usd = buy_usd + sell_usd
        
        # --- MODIFICA RICHIESTA: NORMALIZZAZIONE MANUALE / 10 ---
        # Log10(Vol) diviso per 10. Max teorico 10 Miliardi.
        # Range risultante: 0.0 - 1.0
        log_vol = np.log10(vol_usd + 1)
        features.append(log_vol / 10.0) # Feat 7
        
        # CVD (Range -1 a 1)
        features.append((buy_usd - sell_usd) / vol_usd if vol_usd > 0 else 0) # Feat 8
        
        # Iceberg (Log naturale)
        # Questi rimangono grezzi (log1p) perché raramente superano 3-4
        # Il Trainer gestirà questi piccoli sbalzi
        features.append(np.log1p(sell_usd / (bids_usd[0] + 1))) # Feat 9
        features.append(np.log1p(buy_usd / (asks_usd[0] + 1))) # Feat 10
        
        # --- MACRO ---
        closes = [c[4] for c in ohlcv]
        rsi_norm, _ = calculate_indicators(closes)
        
        # Calcolo distanza SMA semplice inline
        sma = pd.Series(closes).rolling(14).mean().iloc[-1]
        dist_sma = (closes[-1] - sma) / sma if sma > 0 else 0
        
        # Calcolo Candle Change
        last_candle = ohlcv[-2]
        candle_change = (last_candle[4] - last_candle[1]) / last_candle[1]
        
        features.append(rsi_norm)      # Feat 11 (0-1)
        features.append(dist_sma)      # Feat 12 (Piccolo float)
        features.append(candle_change) # Feat 13 (Piccolo float)
        
        return features, ticker['last']
        
    except Exception as e:
        return None, None

print(f"--- ⛏️ MINER V8: Manual Normalization (/10) ---")

# Colonne CSV
cols = [
    'obi_1', 'obi_2', 'obi_3', 'obi_near', 'obi_far',        
    'bid_conc', 'ask_conc',                                  
    'log_vol_norm', 'cvd',                                        
    'bid_ice', 'ask_ice',                                    
    'rsi', 'dist_sma', 'candle_change',                      
    'momentum_1s',
    'target'                                                 
]

if not os.path.exists(FILE_NAME):
    pd.DataFrame(columns=cols).to_csv(FILE_NAME, index=False)

buffer = []
prev_loop_price = None 

while True:
    feats, current_price = get_normalized_features()
    
    if not feats:
        time.sleep(1)
        continue
        
    current_time = time.time()
    
    # Calcolo Momentum
    if prev_loop_price is None:
        momentum = 0.0 
    else:
        momentum = (current_price - prev_loop_price) / prev_loop_price
        
    prev_loop_price = current_price
    feats.append(momentum) # Feat 14 (Totale 15 inputs)
    
    # Buffer
    buffer.append({'f': feats, 'p': current_price, 't': current_time})
    
    # Process Buffer
    for item in buffer[:]:
        if current_time - item['t'] >= LOOK_AHEAD:
            
            past_price_abs = item['p']
            future_price_abs = current_price 
            
            # Target
            is_profitable = 1 if future_price_abs > (past_price_abs * (1 + TARGET_PROFIT)) else 0
            
            # Save
            row = item['f'] + [is_profitable]
            df_row = pd.DataFrame([row], columns=cols)
            df_row.to_csv(FILE_NAME, mode='a', header=False, index=False)
            
            # Print (Mostriamo il LogVol normalizzato per verifica)
            vol_norm = item['f'][7] 
            res = "WIN 🟢" if is_profitable else "NO  🔴"
            print(f"{res} | Vol(0-1): {vol_norm:.2f} | Ice: {item['f'][9]:.2f}", end='\r')
            
            buffer.remove(item)
            
    time.sleep(1)