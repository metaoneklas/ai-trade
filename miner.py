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
from typing import List, Dict, Any

# ================= CONFIGURAZIONE V15 (OFI + ROLLING NORM) =================
SYMBOL = "BNB-USDT"
FILE_NAME = 'training_data_v15.csv'
LOOK_AHEAD = 20             # Secondi per il target
LOOK_BACK = 1.0             # Finestra Momentum
MIN_SAMPLE_INTERVAL = 0.5   # 2Hz
FORCE_SAMPLE_INTERVAL = 2.0 
ROLLING_WINDOW_NORM = 300   # Finestra per la normalizzazione (es. 300 samples = ~2.5 min a 2Hz)
BATCH_SIZE = 10
PRICE_TOL = 1e-9
# ===========================================================================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_book = None
        self.trade_bucket: List[Dict[str, Any]] = []

    def update_book(self, book_data: Dict[str, Any]):
        with self.lock:
            if book_data:
                self.latest_book = dict(book_data)
                self.latest_book['_rcv'] = time.time()

    def add_trades(self, trade_list: List[Dict[str, Any]]):
        with self.lock:
            if trade_list:
                for t in trade_list:
                    try:
                        side = 'sell' if t.get('m', False) else 'buy'
                        self.trade_bucket.append({
                            'p': float(t['p']), 'q': float(t['q']), 'side': side, 't': t.get('T', time.time())
                        })
                    except: continue

    def pop_snapshot(self):
        with self.lock:
            if not self.latest_book: return None, []
            book = dict(self.latest_book)
            trades = self.trade_bucket[:]
            self.trade_bucket = []
            return book, trades

shared = SharedState()

class BingXSocket(threading.Thread):
    def __init__(self, url, channel):
        super().__init__()
        self.url = url
        self.channel = channel
        self.keep_running = True
        self.daemon = True

    def on_message(self, ws, message):
        try:
            if isinstance(message, (bytes, bytearray)):
                utf8_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode='rb').read().decode('utf-8')
            else: utf8_data = message
            if utf8_data == "Ping": ws.send("Pong"); return
            data = json.loads(utf8_data)
            if 'data' in data and data['data']:
                if "trade" in self.channel['dataType']: shared.add_trades(data['data'])
                elif "depth" in self.channel['dataType']: shared.update_book(data['data'])
        except: pass

    def run(self):
        while self.keep_running:
            try:
                websocket.WebSocketApp(self.url, on_open=lambda ws: ws.send(json.dumps(self.channel)), 
                                     on_message=self.on_message).run_forever()
            except: time.sleep(3)

def init_csv(filename, cols):
    if not os.path.exists(filename):
        pd.DataFrame(columns=cols).to_csv(filename, index=False)
        print(f"📁 Created file: {filename}")

def flush_batch(filename, batch_data, cols):
    if not batch_data: return
    df = pd.DataFrame(batch_data, columns=cols)
    df.to_csv(filename, mode='a', header=False, index=False, float_format='%.8f')

# --- CLASSE PER LA NORMALIZZAZIONE ROLLING ---
class RollingNormalizer:
    def __init__(self, window_size):
        self.window = window_size
        self.buffers = {} # Dizionario di deques
        self.stats = {}   # Cache di media e std

    def update(self, key, value):
        if key not in self.buffers:
            self.buffers[key] = deque(maxlen=self.window)
        
        self.buffers[key].append(value)
        
        # Ricalcola stats (o ottimizza con Welford se necessario, qui usiamo numpy su window piccola)
        if len(self.buffers[key]) > 10: # Aspetta di avere un po' di dati
            arr = np.array(self.buffers[key])
            mean = np.mean(arr)
            std = np.std(arr)
            self.stats[key] = (mean, std)
        else:
            self.stats[key] = (value, 1.0) # Fallback iniziale

    def normalize(self, key, value):
        # Z-Score: (x - mean) / std
        if key in self.stats and self.stats[key][1] > 1e-9:
            mean, std = self.stats[key]
            # Clip a +/- 5 sigma per evitare outlier estremi
            return np.clip((value - mean) / std, -5.0, 5.0)
        return 0.0 # Default se non abbiamo storico sufficiente

# ------------------ MAIN MINER V15 ------------------
def main_miner_v15():
    URL = "wss://open-api-swap.bingx.com/swap-market"
    BingXSocket(URL, {"id": "t", "reqType": "sub", "dataType": f"{SYMBOL}@trade"}).start()
    BingXSocket(URL, {"id": "d", "reqType": "sub", "dataType": f"{SYMBOL}@depth5@200ms"}).start()

    # Nuove Feature Normalizzate + OFI
    cols = [
        'z_obi', 'z_ofi', 'z_vol', 'z_spread', 'z_micro_div', # Normalized Features
        'raw_price', 'target_return_bps'
    ]
    init_csv(FILE_NAME, cols)
    
    print(f"--- ⛏️ MINER V15: OFI + ROLLING NORM ({SYMBOL}) ---")
    
    buffer = deque()
    write_queue = []
    history_prices = deque(maxlen=1000)
    
    # Normalizer
    normalizer = RollingNormalizer(ROLLING_WINDOW_NORM)

    # State Variables
    prev_book_snap = None
    last_sample_time = time.time()
    accum_vol = 0.0
    accum_ofi = 0.0 # Accumulatore per OFI tra i sample

    while True:
        cycle_start = time.time()
        current_book, recent_trades = shared.pop_snapshot()

        if not current_book:
            time.sleep(0.001); continue

        try:
            bids = current_book.get('bids', []); asks = current_book.get('asks', [])
            if not bids or not asks: continue
            
            best_bid_p = float(bids[0][0]); best_bid_q = float(bids[0][1])
            best_ask_p = float(asks[0][0]); best_ask_q = float(asks[0][1])
            sum_bid5 = sum([float(x[1]) for x in bids[:5]])
            sum_ask5 = sum([float(x[1]) for x in asks[:5]])
        except: continue

        mid_price = (best_bid_p + best_ask_p) / 2.0
        curr_time = time.time()

        # --- 1. CALCOLO OFI (Physics of Price Discovery) ---
        # Formula di Cont et al.: e_t = e_t^b - e_t^a
        # Calcoliamo l'OFI istantaneo rispetto allo snapshot precedente (anche se è passato 1ms)
        step_ofi = 0.0
        if prev_book_snap:
            # --- Bid Side OFI ---
            e_b = 0.0
            if best_bid_p > prev_book_snap['bid_p']:      # Price Increased -> New Liquidity Added
                e_b = best_bid_q
            elif best_bid_p < prev_book_snap['bid_p']:    # Price Dropped -> Liquidity Removed/Hit
                e_b = -prev_book_snap['bid_q']
            else:                                         # Price Same -> Size Change
                e_b = best_bid_q - prev_book_snap['bid_q']
            
            # --- Ask Side OFI ---
            e_a = 0.0
            if best_ask_p < prev_book_snap['ask_p']:      # Price Dropped (Improved) -> New Liquidity Added
                e_a = best_ask_q
            elif best_ask_p > prev_book_snap['ask_p']:    # Price Increased -> Liquidity Removed/Hit
                e_a = -prev_book_snap['ask_q']
            else:
                e_a = best_ask_q - prev_book_snap['ask_q']
            
            # Net OFI
            step_ofi = e_b - e_a
            accum_ofi += step_ofi

        # --- 2. ACCUMULO VOLUMI ---
        tick_vol = sum([t['p']*t['q'] for t in recent_trades])
        accum_vol += tick_vol

        # Salva stato per il prossimo ciclo (Micro-Ciclo)
        prev_book_snap = {'bid_p': best_bid_p, 'bid_q': best_bid_q, 'ask_p': best_ask_p, 'ask_q': best_ask_q}
        history_prices.append({'t': curr_time, 'p': mid_price})

        # --- 3. CAMPIONAMENTO (2Hz) ---
        time_elapsed = curr_time - last_sample_time
        # OBI corrente
        raw_obi = (sum_bid5 - sum_ask5) / (sum_bid5 + sum_ask5) if (sum_bid5 + sum_ask5) > 0 else 0.0
        
        # Trigger: Attività o Timeout
        has_activity = (accum_vol > 0) or (abs(accum_ofi) > 0.1) # Usa accum_ofi come trigger
        force_sample = time_elapsed > FORCE_SAMPLE_INTERVAL
        
        if (time_elapsed > MIN_SAMPLE_INTERVAL) and (has_activity or force_sample):
            
            # --- Calcolo Feature Grezze ---
            raw_vol = np.log10(accum_vol + 1)
            raw_spread = best_ask_p - best_bid_p
            
            denom = best_bid_q + best_ask_q
            micro_price = (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / denom if denom > 0 else mid_price
            raw_micro_div = micro_price - mid_price

            # --- NORMALIZZAZIONE ONLINE (Rolling Z-Score) ---
            # Prima aggiorniamo le statistiche con i nuovi dati grezzi
            normalizer.update('obi', raw_obi)
            normalizer.update('ofi', accum_ofi)
            normalizer.update('vol', raw_vol)
            normalizer.update('spread', raw_spread)
            normalizer.update('micro_div', raw_micro_div)
            
            # Poi otteniamo il valore normalizzato
            z_obi = normalizer.normalize('obi', raw_obi)
            z_ofi = normalizer.normalize('ofi', accum_ofi)
            z_vol = normalizer.normalize('vol', raw_vol)
            z_spread = normalizer.normalize('spread', raw_spread)
            z_micro_div = normalizer.normalize('micro_div', raw_micro_div)

            feats = [z_obi, z_ofi, z_vol, z_spread, z_micro_div, mid_price]

            # Buffer
            buffer.append({
                'f': feats, 
                'p_entry': mid_price, 
                't': curr_time
            })

            # Reset Accumulatori
            last_sample_time = curr_time
            accum_vol = 0.0
            accum_ofi = 0.0
            
            print(f"✅ V15 | Z-OFI:{z_ofi:.2f} | Z-OBI:{z_obi:.2f} | Z-Vol:{z_vol:.2f}", end='\r')

        # --- 4. TARGET (FIFO) ---
        while len(buffer) > 0:
            head = buffer[0]
            if curr_time - head['t'] >= LOOK_AHEAD:
                # Target in BPS
                future_return = (mid_price - head['p_entry']) / head['p_entry']
                target_bps = future_return * 10000 
                
                # Salviamo: Features Normalizzate + Prezzo Entry (per debug) + Target
                final_row = head['f'][:-1] + [head['p_entry']] + [target_bps]
                write_queue.append(final_row)
                buffer.popleft()
            else:
                break

        # --- 5. SCRITTURA ---
        if len(write_queue) >= BATCH_SIZE:
            flush_batch(FILE_NAME, write_queue, cols)
            write_queue = []

        elapsed = time.time() - cycle_start
        if elapsed < 0.001: time.sleep(0.001)

if __name__ == "__main__":
    try:
        main_miner_v15()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")