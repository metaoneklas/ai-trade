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

# ================= CONFIGURAZIONE V11 (CLEAN & STABLE) =================
SYMBOL = "BNB-USDT"
FILE_NAME = 'training_data_neural.csv'
LOOK_AHEAD = 20             # Target futuro (secondi)
LOOK_BACK = 1.0             # Finestra Momentum (secondi)
MIN_SAMPLE_INTERVAL = 1   # Campionamento max 10Hz (evita ridondanza)
ROLLING_WINDOW = 50         # Aumentato: Memoria degli ultimi 50 trade attivi
PRICE_TOL = 1e-9
# =======================================================================

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

def append_row(filename, row, cols):
    pd.DataFrame([row], columns=cols).to_csv(filename, mode='a', header=False, index=False, float_format='%.10f')

# ------------------ MAIN MINER V11 ------------------
def main_miner_clean():
    URL = "wss://open-api-swap.bingx.com/swap-market"
    BingXSocket(URL, {"id": "t", "reqType": "sub", "dataType": f"{SYMBOL}@trade"}).start()
    BingXSocket(URL, {"id": "d", "reqType": "sub", "dataType": f"{SYMBOL}@depth5@200ms"}).start()

    cols = [
        'obi', 'd_obi', 'log_vol', 'roll_cvd', 'd_cvd', 
        'ice_bid', 'ice_ask', 'spread', 'micro_div', 
        'past_ret', 'mid_price', 'future_ret'
    ]
    init_csv(FILE_NAME, cols)
    
    print(f"--- ⛏️ MINER V11: CLEAN & STABLE ({SYMBOL}) ---")
    print("Logica CVD: Event-Based (Non va a zero nel silenzio)")
    
    buffer = deque(maxlen=5000)
    history_prices = deque(maxlen=200)
    
    # CVD State
    cvd_deque = deque(maxlen=ROLLING_WINDOW)
    prev_cvd = 0.0
    
    prev_state = {'obi': 0.0}
    prev_book_snap = None
    last_sample_time = 0

    while True:
        cycle_start = time.time()
        current_book, recent_trades = shared.pop_snapshot()

        # Attesa dati iniziale
        if not current_book:
            time.sleep(0.1); continue

        try:
            bids = current_book.get('bids', []); asks = current_book.get('asks', [])
            if not bids or not asks: continue
            
            best_bid_p = float(bids[0][0]); best_bid_q = float(bids[0][1])
            best_ask_p = float(asks[0][0]); best_ask_q = float(asks[0][1])
            sum_bid5 = sum([float(x[1]) for x in bids[:5]])
            sum_ask5 = sum([float(x[1]) for x in asks[:5]])
        except: continue

        # --- FEATURES ---
        mid_price = (best_bid_p + best_ask_p) / 2.0
        spread = best_ask_p - best_bid_p
        history_prices.append({'t': time.time(), 'p': mid_price})

        # 1. OBI
        obi = (sum_bid5 - sum_ask5) / (sum_bid5 + sum_ask5) if (sum_bid5 + sum_ask5) > 0 else 0.0
        d_obi = obi - prev_state['obi']

        # 2. Volume & CVD (FIXED LOGIC)
        buy_vol = sum([t['p']*t['q'] for t in recent_trades if t['side']=='buy'])
        sell_vol = sum([t['p']*t['q'] for t in recent_trades if t['side']=='sell'])
        tick_vol = buy_vol + sell_vol
        net_vol = buy_vol - sell_vol

        # FIX: Aggiorniamo la deque SOLO se c'è volume. 
        # Se il mercato tace, manteniamo la memoria della pressione precedente.
        if tick_vol > 0:
            cvd_deque.append(net_vol)
        
        current_roll_cvd = sum(cvd_deque) # Somma degli ultimi 50 trade ATTIVI
        d_cvd = current_roll_cvd - prev_cvd
        
        if tick_vol > 0: # Aggiorniamo prev_cvd solo se è cambiato qualcosa
            prev_cvd = current_roll_cvd

        # 3. Iceberg
        ice_bid = 0.0; ice_ask = 0.0
        if prev_book_snap:
            if abs(best_bid_p - prev_book_snap['bid_p']) < PRICE_TOL:
                delta = prev_book_snap['bid_q'] - best_bid_q
                exec_vol = sum([t['q'] for t in recent_trades if t['side']=='sell' and abs(t['p']-best_bid_p)<PRICE_TOL])
                if (exec_vol - delta) > 0.0001: ice_bid = exec_vol - delta
            
            if abs(best_ask_p - prev_book_snap['ask_p']) < PRICE_TOL:
                delta = prev_book_snap['ask_q'] - best_ask_q
                exec_vol = sum([t['q'] for t in recent_trades if t['side']=='buy' and abs(t['p']-best_ask_p)<PRICE_TOL])
                if (exec_vol - delta) > 0.0001: ice_ask = exec_vol - delta

        # 4. MicroPrice & Momentum
        denom = best_bid_q + best_ask_q
        micro_price = (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / denom if denom > 0 else mid_price
        micro_div = micro_price - mid_price

        past_ret = 0.0
        curr_time = time.time()
        for h in history_prices:
            if curr_time - h['t'] <= LOOK_BACK:
                past_ret = (mid_price - h['p']) / h['p']
                break

        # Update States
        prev_state['obi'] = obi
        prev_book_snap = {'bid_p': best_bid_p, 'bid_q': best_bid_q, 'ask_p': best_ask_p, 'ask_q': best_ask_q}

        # --- BUFFERING ---
        if (curr_time - last_sample_time) > MIN_SAMPLE_INTERVAL:
            last_sample_time = curr_time
            
            feats = [
                obi, d_obi, 
                np.log10(tick_vol+1), 
                current_roll_cvd, 
                d_cvd,
                ice_bid, ice_ask, 
                spread, micro_div, 
                past_ret, 
                mid_price # Serve per calcolo target, verrà tolta nel training
            ]
            buffer.append({'f': feats, 'p': mid_price, 't': curr_time})
            
            # Clean Status Print
            print(f"🟢 RUNNING | P:{mid_price:.2f} | CVD:{current_roll_cvd:.0f} | Buffer:{len(buffer)}   ", end='\r')

        # --- TARGET & WRITE ---
        now = time.time()
        # Copia sicura per iterare e rimuovere
        pending = list(buffer)
        
        for item in pending:
            if now - item['t'] >= LOOK_AHEAD:
                past_p = item['p']
                curr_p = mid_price
                # Calcolo Target
                ret_pct = (curr_p - past_p) / past_p
                
                # Sostituiamo l'ultimo elemento (mid_price) con il target (future_ret)
                # O meglio: aggiungiamo il target alla fine
                # La lista 'feats' ha mid_price in ultima posizione [-1]
                # Salviamo tutto per sicurezza, poi pandas filtra
                row = item['f'] + [ret_pct]
                
                append_row(FILE_NAME, row, cols)
                buffer.remove(item)

        elapsed = time.time() - cycle_start
        if elapsed < 0.05: time.sleep(0.05)

if __name__ == "__main__":
    try:
        main_miner_clean()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")