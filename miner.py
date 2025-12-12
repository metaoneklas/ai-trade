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

# ================= CONFIGURAZIONE V12 (MAX GAIN TARGET) =================
SYMBOL = "BNB-USDT"
FILE_NAME = 'training_data_max_gain.csv'
LOOK_AHEAD = 20             # Finestra temporale (secondi)
LOOK_BACK = 1.0             # Finestra Momentum (secondi)
MIN_SAMPLE_INTERVAL = 1     # Campionamento max 10Hz
ROLLING_WINDOW = 50         # Memoria CVD
BATCH_SIZE = 50             # Scrittura su disco ogni 50 righe (salva CPU/IO)
PRICE_TOL = 1e-9
# ========================================================================

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

# --- BATCH WRITER PER PERFORMANCE ---
def flush_batch(filename, batch_data, cols):
    if not batch_data: return
    df = pd.DataFrame(batch_data, columns=cols)
    df.to_csv(filename, mode='a', header=False, index=False, float_format='%.10f')

# ------------------ MAIN MINER V12 ------------------
def main_miner_max_gain():
    URL = "wss://open-api-swap.bingx.com/swap-market"
    BingXSocket(URL, {"id": "t", "reqType": "sub", "dataType": f"{SYMBOL}@trade"}).start()
    BingXSocket(URL, {"id": "d", "reqType": "sub", "dataType": f"{SYMBOL}@depth5@200ms"}).start()

    # Features + Target (max_gain)
    cols = [
        'obi', 'd_obi', 'log_vol', 'roll_cvd', 'd_cvd', 
        'ice_bid', 'ice_ask', 'spread', 'micro_div', 
        'past_ret', 'mid_price', 'target_max_gain' 
    ]
    init_csv(FILE_NAME, cols)
    
    print(f"--- ⛏️ MINER V12: MAX GAIN TARGET ({SYMBOL}) ---")
    print(f"Target: Massimo rialzo % nei successivi {LOOK_AHEAD}s")
    
    # Strutture dati
    buffer = deque()            # Contiene i sample in attesa di maturazione
    write_queue = []            # Contiene i dati pronti per il CSV (Batching)
    history_prices = deque(maxlen=200)
    cvd_deque = deque(maxlen=ROLLING_WINDOW)
    
    # Variabili stato
    prev_cvd_sum = 0.0
    prev_state = {'obi': 0.0}
    prev_book_snap = None
    last_sample_time = 0

    while True:
        cycle_start = time.time()
        current_book, recent_trades = shared.pop_snapshot()

        if not current_book:
            time.sleep(0.01)
            continue

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
        
        # --- 1. AGGIORNAMENTO MAX_PRICE NEL BUFFER (CRUCIALE) ---
        # Ogni elemento nel buffer "vede" il prezzo attuale.
        # Se il prezzo attuale è > del massimo visto finora, aggiorniamo.
        for item in buffer:
            if mid_price > item['max_p']:
                item['max_p'] = mid_price

        # --- 2. CALCOLO FEATURES ---
        spread = best_ask_p - best_bid_p
        history_prices.append({'t': curr_time, 'p': mid_price})

        # OBI
        obi = (sum_bid5 - sum_ask5) / (sum_bid5 + sum_ask5) if (sum_bid5 + sum_ask5) > 0 else 0.0
        d_obi = obi - prev_state['obi']

        # CVD Efficiente
        buy_vol = sum([t['p']*t['q'] for t in recent_trades if t['side']=='buy'])
        sell_vol = sum([t['p']*t['q'] for t in recent_trades if t['side']=='sell'])
        tick_vol = buy_vol + sell_vol
        net_vol = buy_vol - sell_vol

        if tick_vol > 0:
            if len(cvd_deque) == ROLLING_WINDOW:
                removed = cvd_deque.popleft()
                prev_cvd_sum -= removed # Aggiornamento incrementale
            cvd_deque.append(net_vol)
            prev_cvd_sum += net_vol
        
        d_cvd = prev_cvd_sum - (prev_cvd_sum - net_vol if tick_vol > 0 else prev_cvd_sum)

        # Iceberg Logic
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

        # MicroPrice & Momentum
        denom = best_bid_q + best_ask_q
        micro_price = (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / denom if denom > 0 else mid_price
        micro_div = micro_price - mid_price

        past_ret = 0.0
        for h in history_prices:
            if curr_time - h['t'] <= LOOK_BACK:
                past_ret = (mid_price - h['p']) / h['p']
                break

        prev_state['obi'] = obi
        prev_book_snap = {'bid_p': best_bid_p, 'bid_q': best_bid_q, 'ask_p': best_ask_p, 'ask_q': best_ask_q}

        # --- 3. CREAZIONE NUOVO SAMPLE ---
        if (curr_time - last_sample_time) > MIN_SAMPLE_INTERVAL:
            last_sample_time = curr_time
            
            feats = [
                obi, d_obi, 
                np.log10(tick_vol+1), 
                prev_cvd_sum, 
                d_cvd,
                ice_bid, ice_ask, 
                spread, micro_div, 
                past_ret, 
                mid_price 
            ]
            
            # Inizializziamo 'max_p' con il prezzo attuale
            buffer.append({
                'f': feats, 
                'p_entry': mid_price, 
                't': curr_time, 
                'max_p': mid_price # High Watermark iniziale
            })

            print(f"🟢 BUFFER:{len(buffer)} | WRITE_Q:{len(write_queue)} | P:{mid_price:.2f}   ", end='\r')

        # --- 4. CONTROLLO TARGET MATURATI (FIFO) ---
        # Controlliamo solo la testa della coda (efficienza massima)
        while len(buffer) > 0:
            head_item = buffer[0]
            if curr_time - head_item['t'] >= LOOK_AHEAD:
                # Il tempo è scaduto. Il target è il MAX raggiunto diviso l'entry
                max_gain_pct = (head_item['max_p'] - head_item['p_entry']) / head_item['p_entry']
                
                # Sostituiamo l'ultima colonna (mid_price entry) con il target
                final_row = head_item['f'][:-1] + [head_item['p_entry']] + [max_gain_pct] 
                # Nota: ho mantenuto mid_price e aggiunto il target alla fine per chiarezza,
                # ma nel 'cols' sopra ho messo 'mid_price' e 'target_max_gain', quindi siamo allineati.
                
                write_queue.append(final_row)
                buffer.popleft() # Rimuovi dalla testa
            else:
                break # Se il primo non è maturo, neanche gli altri lo sono

        # --- 5. BATCH WRITING ---
        if len(write_queue) >= BATCH_SIZE:
            flush_batch(FILE_NAME, write_queue, cols)
            write_queue = []

        elapsed = time.time() - cycle_start
        if elapsed < 0.01: time.sleep(0.01)

if __name__ == "__main__":
    try:
        main_miner_max_gain()
    except KeyboardInterrupt:
        print("\n🛑 Stopped. Saving remaining data...")
        # Qui potresti aggiungere un flush finale se vuoi salvare i dati parziali