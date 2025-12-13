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

# ================= CONFIGURAZIONE V14 (FINAL DATA SCIENCE) =================
SYMBOL = "BNB-USDT"
FILE_NAME = 'training_data_final.csv'
LOOK_AHEAD = 20             # Orizzonte temporale (secondi) per il target
LOOK_BACK = 1.0             # Finestra Momentum (secondi)
MIN_SAMPLE_INTERVAL = 0.5   # Frequenza base (2Hz)
FORCE_SAMPLE_INTERVAL = 2.0 # Scrivi comunque ogni 2s anche se mercato fermo
ROLLING_WINDOW = 50         # Finestra per il CVD Rolling
BATCH_SIZE = 10             # Scrittura su disco a blocchi
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
                            'p': float(t['p']), 
                            'q': float(t['q']), 
                            'side': side, 
                            't': t.get('T', time.time())
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

# ------------------ MAIN MINER V14 (FINAL) ------------------
def main_miner_final():
    URL = "wss://open-api-swap.bingx.com/swap-market"
    BingXSocket(URL, {"id": "t", "reqType": "sub", "dataType": f"{SYMBOL}@trade"}).start()
    BingXSocket(URL, {"id": "d", "reqType": "sub", "dataType": f"{SYMBOL}@depth5@200ms"}).start()

    # Features Aggiornate
    # Nota: 'target_return_bps' sostituisce 'target_max_gain'
    cols = [
        'obi', 'd_obi', 'log_vol', 'roll_cvd', 'd_cvd', 
        'ice_bid', 'ice_ask', 'spread', 'micro_div', 
        'past_ret', 'mid_price', 'target_return_bps' 
    ]
    init_csv(FILE_NAME, cols)
    
    print(f"--- ⛏️ MINER V14: SMART SAMPLING & SYMMETRIC TARGET ({SYMBOL}) ---")
    
    # Strutture dati
    buffer = deque()
    write_queue = []
    history_prices = deque(maxlen=1000) 
    cvd_deque = deque(maxlen=ROLLING_WINDOW)
    
    # Stato persistente
    prev_cvd_sum = 0.0
    prev_book_snap = None
    last_sample_time = time.time()
    
    # Accumulatori (reset ad ogni sample)
    accum_vol = 0.0
    accum_ice_bid = 0.0
    accum_ice_ask = 0.0
    
    # Valori di riferimento (aggiornati ad ogni sample)
    last_stored_obi = 0.0
    last_stored_cvd = 0.0

    while True:
        cycle_start = time.time()
        current_book, recent_trades = shared.pop_snapshot()

        if not current_book:
            time.sleep(0.001)
            continue

        try:
            bids = current_book.get('bids', [])
            asks = current_book.get('asks', [])
            if not bids or not asks: continue
            
            best_bid_p = float(bids[0][0]); best_bid_q = float(bids[0][1])
            best_ask_p = float(asks[0][0]); best_ask_q = float(asks[0][1])
            
            sum_bid5 = sum([float(x[1]) for x in bids[:5]])
            sum_ask5 = sum([float(x[1]) for x in asks[:5]])
        except: continue

        mid_price = (best_bid_p + best_ask_p) / 2.0
        curr_time = time.time()

        # --- 1. PROCESSING AD ALTA FREQUENZA (Accumulo) ---
        
        # CVD & Volume Accumulato
        buy_vol = sum([t['p']*t['q'] for t in recent_trades if t['side']=='buy'])
        sell_vol = sum([t['p']*t['q'] for t in recent_trades if t['side']=='sell'])
        tick_vol = buy_vol + sell_vol
        net_vol = buy_vol - sell_vol
        
        accum_vol += tick_vol # Accumuliamo il volume tra i sample

        if tick_vol > 0:
            if len(cvd_deque) == ROLLING_WINDOW:
                removed = cvd_deque.popleft()
                prev_cvd_sum -= removed
            cvd_deque.append(net_vol)
            prev_cvd_sum += net_vol

        # Iceberg Logic (Accumulativa)
        if prev_book_snap:
            # Bid side
            if abs(best_bid_p - prev_book_snap['bid_p']) < PRICE_TOL:
                visible_delta = prev_book_snap['bid_q'] - best_bid_q
                exec_sell_vol = sum([t['q'] for t in recent_trades if t['side']=='sell' and abs(t['p']-best_bid_p)<PRICE_TOL])
                hidden_exec = exec_sell_vol - visible_delta
                if hidden_exec > 0.0001: accum_ice_bid += hidden_exec

            # Ask side
            if abs(best_ask_p - prev_book_snap['ask_p']) < PRICE_TOL:
                visible_delta = prev_book_snap['ask_q'] - best_ask_q
                exec_buy_vol = sum([t['q'] for t in recent_trades if t['side']=='buy' and abs(t['p']-best_ask_p)<PRICE_TOL])
                hidden_exec = exec_buy_vol - visible_delta
                if hidden_exec > 0.0001: accum_ice_ask += hidden_exec

        prev_book_snap = {'bid_p': best_bid_p, 'bid_q': best_bid_q, 'ask_p': best_ask_p, 'ask_q': best_ask_q}
        history_prices.append({'t': curr_time, 'p': mid_price})

        # --- 2. LOGICA DI SMART SAMPLING ---
        # Calcoliamo l'OBI corrente per vedere se è cambiato significativamente
        current_obi = (sum_bid5 - sum_ask5) / (sum_bid5 + sum_ask5) if (sum_bid5 + sum_ask5) > 0 else 0.0
        
        # Trigger: C'è stato volume? O l'OBI è cambiato molto? O è passato troppo tempo?
        time_elapsed = curr_time - last_sample_time
        has_activity = (accum_vol > 0) or (abs(current_obi - last_stored_obi) > 0.05)
        force_sample = time_elapsed > FORCE_SAMPLE_INTERVAL
        
        is_sample_time = (time_elapsed > MIN_SAMPLE_INTERVAL) and (has_activity or force_sample)

        if is_sample_time:
            
            # --- 3. CALCOLO FEATURES ---
            d_obi = current_obi - last_stored_obi
            d_cvd = prev_cvd_sum - last_stored_cvd
            spread = best_ask_p - best_bid_p
            
            denom = best_bid_q + best_ask_q
            micro_price = (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / denom if denom > 0 else mid_price
            micro_div = micro_price - mid_price

            # Momentum (Past Return)
            target_ts = curr_time - LOOK_BACK
            found_p = history_prices[0]['p']
            for i in range(len(history_prices)-1, -1, -1):
                if history_prices[i]['t'] <= target_ts:
                    found_p = history_prices[i]['p']; break
            past_ret = (mid_price - found_p) / found_p if found_p > 0 else 0.0

            feats = [
                current_obi, 
                d_obi, 
                np.log10(accum_vol + 1), 
                prev_cvd_sum, 
                d_cvd,
                accum_ice_bid, 
                accum_ice_ask, 
                spread, 
                micro_div, 
                past_ret, 
                mid_price 
            ]

            # Inseriamo nel buffer temporale (Entry Price è mid_price attuale)
            buffer.append({
                'f': feats, 
                'p_entry': mid_price, 
                't': curr_time
            })

            # Reset Accumulatori e Update Riferimenti
            last_sample_time = curr_time
            last_stored_obi = current_obi
            last_stored_cvd = prev_cvd_sum
            accum_vol = 0.0
            accum_ice_bid = 0.0
            accum_ice_ask = 0.0

            print(f"✅ RECORDED | Buff:{len(buffer)} | OBI:{current_obi:.2f} | Vol:{feats[2]:.2f} ", end='\r')

        # --- 4. CALCOLO TARGET (FIFO) - FUTURE RETURN ---
        while len(buffer) > 0:
            head = buffer[0]
            if curr_time - head['t'] >= LOOK_AHEAD:
                # Target: Ritorno in Basis Points (BPS) dopo LOOK_AHEAD secondi
                # Se il prezzo sale dell'1%, il valore sarà 100. Se scende, sarà negativo.
                future_return = (mid_price - head['p_entry']) / head['p_entry']
                target_bps = future_return * 10000 
                
                # Sostituiamo l'ultima feature (mid_price entry) con il target finale
                final_row = head['f'][:-1] + [head['p_entry']] + [target_bps]
                
                write_queue.append(final_row)
                buffer.popleft()
            else:
                break

        # --- 5. SCRITTURA SU DISCO ---
        if len(write_queue) >= BATCH_SIZE:
            flush_batch(FILE_NAME, write_queue, cols)
            write_queue = []

        elapsed = time.time() - cycle_start
        if elapsed < 0.001: time.sleep(0.001)

if __name__ == "__main__":
    try:
        main_miner_final()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")