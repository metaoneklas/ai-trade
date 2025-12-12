import websocket
import threading
import time
import json
import gzip
import io
import pandas as pd
import numpy as np
import os
from collections import deque

# ================= CONFIGURAZIONE =================
SYMBOL = "BNB-USDT" 
FILE_NAME = 'training_data_context.csv'
LOOK_AHEAD = 10             # Target 10s
TARGET_PROFIT = 0.0003      # 0.03%
VOL_FILTER = 500            # Filtro Volume ($)
ROLLING_WINDOW = 6          # Quanti cicli da 0.5s accumulare per il CVD (6 * 0.5 = 3 secondi)
# ==================================================

# --- MEMORIA CONDIVISA (Invariata) ---
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_book = None
        self.trade_bucket = []

    def update_book(self, book_data):
        with self.lock:
            if book_data and isinstance(book_data, dict):
                self.latest_book = book_data

    def add_trades(self, trade_list):
        with self.lock:
            if not trade_list or not isinstance(trade_list, list): return
            for t in trade_list:
                try:
                    side = 'sell' if t.get('m', False) else 'buy'
                    self.trade_bucket.append({
                        'p': float(t['p']), 'q': float(t['q']), 'side': side, 't': t['T']
                    })
                except: pass

shared = SharedState()

# --- WEBSOCKET CLASS (Invariata - Robusta) ---
class BingXSocket(threading.Thread):
    def __init__(self, url, channel):
        super().__init__()
        self.url = url
        self.channel = channel
        self.ws = None
        self.keep_running = True
        self.daemon = True 

    def on_open(self, ws):
        print(f"🔌 Connected {self.channel['dataType']}")
        ws.send(json.dumps(self.channel))

    def on_message(self, ws, message):
        try:
            compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode='rb')
            utf8_data = compressed_data.read().decode('utf-8')
            if utf8_data == "Ping":
                ws.send("Pong")
                return
            data = json.loads(utf8_data)
            if 'data' in data and data['data'] is not None:
                dtype = self.channel['dataType']
                if "trade" in dtype: shared.add_trades(data['data'])
                elif "depth" in dtype: shared.update_book(data['data'])
        except: pass

    def run(self):
        while self.keep_running:
            try:
                self.ws = websocket.WebSocketApp(self.url, on_open=self.on_open, on_message=self.on_message)
                self.ws.run_forever()
            except: time.sleep(5)

# --- MAIN LOGIC CON MEMORIA ---
def main_miner():
    URL = "wss://open-api-swap.bingx.com/swap-market"
    BingXSocket(URL, {"id": "t", "reqType": "sub", "dataType": f"{SYMBOL}@trade"}).start()
    BingXSocket(URL, {"id": "d", "reqType": "sub", "dataType": f"{SYMBOL}@depth5@500ms"}).start()

    print("--- ⛏️ MINER V6: Context & Microstructure ---")
    print("Features: [OBI, d_OBI, LogVol, Roll_CVD, Ice_Bid, Ice_Ask, Spread, d_Spread, Micro_Div]")
    time.sleep(3)

    cols = [
        'obi', 'd_obi',         # Imbalance + Variazione
        'log_vol',              # Volume
        'roll_cvd',             # CVD Accumulato 3s
        'ice_bid', 'ice_ask',   # Iceberg
        'spread', 'd_spread',   # Spread + Variazione (Squeeze)
        'micro_div',            # Divergenza Prezzo/MicroPrezzo
        'target'
    ]
    
    if not os.path.exists(FILE_NAME):
        pd.DataFrame(columns=cols).to_csv(FILE_NAME, index=False)

    buffer = []         # Per il salvataggio CSV (Targeting)
    
    # MEMORIA STORICA
    prev_state = None   # Per calcolare i Delta (t vs t-1)
    prev_book_snap = None # Per calcolare Iceberg
    cvd_deque = deque(maxlen=ROLLING_WINDOW) # Per il Rolling CVD

    try:
        while True:
            # 1. Thread-Safe Snapshot
            with shared.lock:
                if shared.latest_book is None:
                    time.sleep(0.5)
                    continue
                current_book = shared.latest_book.copy()
                recent_trades = shared.trade_bucket[:]
                shared.trade_bucket = [] 

            # Parsing
            try:
                bids = current_book.get('bids', [])
                asks = current_book.get('asks', [])
                if not bids or not asks: continue

                best_bid_p = float(bids[0][0])
                best_bid_q = float(bids[0][1])
                best_ask_p = float(asks[0][0])
                best_ask_q = float(asks[0][1])
                
                # Walls (Top 5 levels)
                sum_bid5 = sum([float(x[1]) for x in bids[:5]])
                sum_ask5 = sum([float(x[1]) for x in asks[:5]])
            except: continue

            # --- 2. Feature Calculation ---

            # A. Basic Metrics
            obi = (sum_bid5 - sum_ask5) / (sum_bid5 + sum_ask5) if (sum_bid5 + sum_ask5) > 0 else 0
            spread = best_ask_p - best_bid_p
            
            # B. Volume & Rolling CVD
            buy_vol = sum([t['p'] * t['q'] for t in recent_trades if t['side'] == 'buy'])
            sell_vol = sum([t['p'] * t['q'] for t in recent_trades if t['side'] == 'sell'])
            tick_vol = buy_vol + sell_vol
            tick_net = buy_vol - sell_vol # Netto di questo 0.5s
            
            # Aggiungiamo alla coda rotante
            cvd_deque.append(tick_net)
            
            # Calcoliamo la somma degli ultimi N cicli (es. 3 secondi)
            rolling_net_vol = sum(cvd_deque)
            # Normalizziamo su scala logaritmica/relativa per non avere numeri enormi
            # Usiamo una tanh per tenerlo tra -1 e 1 approssimativamente
            rolling_cvd_norm = np.tanh(rolling_net_vol / 10000) # Assumendo 10k come volume alto

            # C. Iceberg (Come V5)
            ice_bid = 0.0
            ice_ask = 0.0
            if prev_book_snap:
                if best_bid_p == prev_book_snap['bid_p']:
                    delta_v = prev_book_snap['bid_q'] - best_bid_q
                    sell_exec = sum([t['q'] for t in recent_trades if t['side']=='sell'])
                    if sell_exec > delta_v + 0.0001: ice_bid = np.log1p(sell_exec - delta_v)
                
                if best_ask_p == prev_book_snap['ask_p']:
                    delta_v = prev_book_snap['ask_q'] - best_ask_q
                    buy_exec = sum([t['q'] for t in recent_trades if t['side']=='buy'])
                    if buy_exec > delta_v + 0.0001: ice_ask = np.log1p(buy_exec - delta_v)

            # D. Micro-Price Divergence (NEW!)
            # Il prezzo "vero" considerando dove sta il peso del volume
            mid_price = (best_bid_p + best_ask_p) / 2
            # Formula Microprice: (BidP * AskQ + AskP * BidQ) / (BidQ + AskQ)
            denom = best_bid_q + best_ask_q
            if denom > 0:
                micro_price = (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / denom
                # Divergenza: Se micro > mid, pressione buy
                micro_div = (micro_price - mid_price)
            else:
                micro_div = 0

            # E. DELTAS (Variazione rispetto al ciclo precedente)
            if prev_state:
                d_obi = obi - prev_state['obi']
                d_spread = spread - prev_state['spread']
            else:
                d_obi = 0
                d_spread = 0

            # --- Aggiornamento Stati ---
            prev_state = {'obi': obi, 'spread': spread}
            prev_book_snap = {'bid_p': best_bid_p, 'bid_q': best_bid_q, 'ask_p': best_ask_p, 'ask_q': best_ask_q}

            # --- 3. Save Logic ---
            # Filtro: O volume, O iceberg, O cambio forte di OBI
            has_activity = (tick_vol > VOL_FILTER) or (ice_bid > 0.1) or (ice_ask > 0.1) or (abs(d_obi) > 0.2)

            if has_activity:
                feats = [
                    round(obi, 4),
                    round(d_obi, 4),        # NEW: Direzione del muro
                    round(np.log10(tick_vol + 1), 4),
                    round(rolling_cvd_norm, 4), # NEW: Pressione ultimi 3s
                    round(ice_bid, 4),
                    round(ice_ask, 4),
                    round(spread, 2),
                    round(d_spread, 2),     # NEW: Spread Squeeze
                    round(micro_div, 2)     # NEW: Microstructure pressure
                ]
                
                buffer.append({'f': feats, 'p': best_bid_p, 't': time.time()})
                
                # Log contestuale
                trend = "↗️" if d_obi > 0.05 else ("↘️" if d_obi < -0.05 else "➡️")
                print(f"OBI:{obi:.2f} ({trend}) | CVD_3s:{rolling_net_vol/1000:.1f}k | IceB:{ice_bid:.1f}", end='\r')

            else:
                print(f"Listening... {best_bid_p:.1f}", end='\r')

            # --- 4. Target Check ---
            now = time.time()
            for item in buffer[:]:
                if now - item['t'] >= LOOK_AHEAD:
                    is_profit = 1 if best_bid_p > (item['p'] * (1 + TARGET_PROFIT)) else 0
                    row = item['f'] + [is_profit]
                    pd.DataFrame([row], columns=cols).to_csv(FILE_NAME, mode='a', header=False, index=False)
                    buffer.remove(item)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Stopped.")

if __name__ == "__main__":
    main_miner()