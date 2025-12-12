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

# ================= CONFIGURAZIONE =================
SYMBOL = "BNB-USDT"
FILE_NAME = 'training_data_context_v7.csv'
LOOK_AHEAD = 10             # Target 10s
TARGET_PROFIT = 0.0003      # 0.03%
VOL_FILTER = 500            # Filtro Volume ($)
ROLLING_WINDOW = 6          # 6 * 0.5s = 3s
SAMPLE_RATE_CALM = 0.05     # Percentuale di salvataggio in regime calmo (5%)
MAX_BUFFER = 2000           # Dimensione massima del buffer in memoria
LATENCY_THRESHOLD = 0.5     # Scarta pacchetti più vecchi di 0.5s
PRICE_TOL = 1e-8            # tolleranza di confronto prezzi
# ==================================================

# ------------------ SHARED STATE (THREAD-SAFE) ------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_book = None  # dict con book + ts (timestamp in seconds)
        self.trade_bucket: List[Dict[str, Any]] = []

    def update_book(self, book_data: Dict[str, Any]):
        # Aggiunge timestamp di ricezione per controllo latenza
        with self.lock:
            if book_data and isinstance(book_data, dict):
                book_copy = dict(book_data)  # shallow copy
                book_copy['_recv_ts'] = time.time()
                self.latest_book = book_copy

    def add_trades(self, trade_list: List[Dict[str, Any]]):
        with self.lock:
            if not trade_list or not isinstance(trade_list, list):
                return
            for t in trade_list:
                try:
                    side = 'sell' if t.get('m', False) else 'buy'
                    self.trade_bucket.append({
                        'p': float(t['p']),
                        'q': float(t['q']),
                        'side': side,
                        't': t.get('T', time.time())
                    })
                except Exception:
                    # ignora trade malformato
                    continue

    def pop_snapshot(self):
        """Ritorna (book, trades) e svuota trade_bucket in modo atomico."""
        with self.lock:
            book = None if self.latest_book is None else dict(self.latest_book)
            trades = self.trade_bucket[:]
            self.trade_bucket = []
            return book, trades

shared = SharedState()

# ------------------ WEBSOCKET THREAD ------------------
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
        try:
            ws.send(json.dumps(self.channel))
        except Exception:
            pass

    def on_message(self, ws, message):
        try:
            # message può essere bytes compressi
            if isinstance(message, (bytes, bytearray)):
                compressed_data = gzip.GzipFile(fileobj=io.BytesIO(message), mode='rb')
                utf8_data = compressed_data.read().decode('utf-8')
            else:
                utf8_data = message if isinstance(message, str) else str(message)

            if utf8_data == "Ping":
                try:
                    ws.send("Pong")
                except: pass
                return

            data = json.loads(utf8_data)
            if 'data' not in data or data['data'] is None:
                return

            dtype = self.channel['dataType']
            if "trade" in dtype:
                shared.add_trades(data['data'])
            elif "depth" in dtype:
                shared.update_book(data['data'])
        except Exception:
            # non rompere il websocket per errori di parsing
            return

    def run(self):
        while self.keep_running:
            try:
                self.ws = websocket.WebSocketApp(self.url, on_open=self.on_open, on_message=self.on_message)
                self.ws.run_forever()
            except Exception:
                time.sleep(3)

# ------------------ SALVATAGGIO (CSV) ------------------
def ensure_csv_exists(filename: str, cols: List[str]):
    if not os.path.exists(filename):
        pd.DataFrame(columns=cols).to_csv(filename, index=False)

def append_row_csv(filename: str, row: List[Any], cols: List[str]):
    df = pd.DataFrame([row], columns=cols)
    df.to_csv(filename, mode='a', header=False, index=False)

# ------------------ MAIN MINER V7 ------------------
def main_miner_v7():
    URL = "wss://open-api-swap.bingx.com/swap-market"
    BingXSocket(URL, {"id": "t", "reqType": "sub", "dataType": f"{SYMBOL}@trade"}).start()
    BingXSocket(URL, {"id": "d", "reqType": "sub", "dataType": f"{SYMBOL}@depth5@500ms"}).start()

    print("--- ⛏️ MINER V7: Context & Microstructure (Thread-safe + Balanced Sampling) ---")
    print("Features: [OBI, d_OBI, LogVol, Roll_CVD, Ice_Bid, Ice_Ask, Spread, d_Spread, Micro_Div, target]")
    time.sleep(2)

    cols = [
        'obi', 'd_obi',
        'log_vol',
        'roll_cvd',
        'ice_bid', 'ice_ask',
        'spread', 'd_spread',
        'micro_div',
        'target'
    ]
    ensure_csv_exists(FILE_NAME, cols)

    # buffer per target labeling
    buffer = deque(maxlen=MAX_BUFFER)

    prev_state = None
    prev_book_snap = None
    cvd_deque = deque(maxlen=ROLLING_WINDOW)

    try:
        last_loop = time.time()
        while True:
            loop_start = time.time()

            # 1) Prendo snapshot atomico
            current_book, recent_trades = shared.pop_snapshot()

            if current_book is None:
                # se non abbiamo ancora un book, aspetta breve
                time.sleep(0.1)
                continue

            # 1b) controllo latenza del book (se è troppo vecchio, scarto)
            recv_ts = current_book.get('_recv_ts', time.time())
            now = time.time()
            latency = now - recv_ts
            if latency > LATENCY_THRESHOLD:
                # scartalo per evitare datapoint basati su book vecchio
                # mantieni comunque eventuali trades (sono già stati presi)
                # (qui decidiamo di non saltare il ciclo completamente; ma possiamo)
                # Stampa informativa minima
                print(f"Scartato book vecchio (latency={latency:.3f}s)", end='\r')
                # ma continuiamo con il prossimo ciclo
                time.sleep(0.05)
                continue

            # 2) parse book
            try:
                bids = current_book.get('bids', [])
                asks = current_book.get('asks', [])
                if not bids or not asks:
                    time.sleep(0.05)
                    continue

                best_bid_p = float(bids[0][0]); best_bid_q = float(bids[0][1])
                best_ask_p = float(asks[0][0]); best_ask_q = float(asks[0][1])
                sum_bid5 = sum([float(x[1]) for x in bids[:5]])
                sum_ask5 = sum([float(x[1]) for x in asks[:5]])
            except Exception:
                time.sleep(0.05)
                continue

            # --- Feature calculation ---

            # OBI
            obi = (sum_bid5 - sum_ask5) / (sum_bid5 + sum_ask5) if (sum_bid5 + sum_ask5) > 0 else 0.0
            spread = best_ask_p - best_bid_p

            # Volume / CVD
            buy_vol = sum([t['p'] * t['q'] for t in recent_trades if t['side'] == 'buy'])
            sell_vol = sum([t['p'] * t['q'] for t in recent_trades if t['side'] == 'sell'])
            tick_vol = buy_vol + sell_vol
            tick_net = buy_vol - sell_vol
            cvd_deque.append(tick_net)
            rolling_net_vol = sum(cvd_deque)
            rolling_cvd_norm = np.tanh(rolling_net_vol / 10000.0)

            # Iceberg detection migliorata: considero solo trades eseguiti ESATTAMENTE (entro tolleranza) al livello best_bid_p / best_ask_p
            def executed_at_price(trades: List[Dict[str, Any]], price: float, side_match: str = None):
                s = 0.0
                for tt in trades:
                    if abs(tt['p'] - price) <= PRICE_TOL:
                        if side_match is None or tt['side'] == side_match:
                            s += tt['q']
                return s

            ice_bid = 0.0
            ice_ask = 0.0
            if prev_book_snap is not None:
                # lato bid
                if abs(best_bid_p - prev_book_snap['bid_p']) <= PRICE_TOL:
                    delta_v = prev_book_snap['bid_q'] - best_bid_q
                    # esecuzioni venditrice a quel prezzo (sell at bid)
                    sell_exec = executed_at_price(recent_trades, best_bid_p, side_match='sell')
                    if sell_exec > max(0.0, delta_v) + 1e-9:
                        ice_bid = np.log1p(max(0.0, sell_exec - max(0.0, delta_v)))

                # lato ask
                if abs(best_ask_p - prev_book_snap['ask_p']) <= PRICE_TOL:
                    delta_v = prev_book_snap['ask_q'] - best_ask_q
                    buy_exec = executed_at_price(recent_trades, best_ask_p, side_match='buy')
                    if buy_exec > max(0.0, delta_v) + 1e-9:
                        ice_ask = np.log1p(max(0.0, buy_exec - max(0.0, delta_v)))

            # Micro-price divergence
            mid_price = (best_bid_p + best_ask_p) / 2.0
            denom = best_bid_q + best_ask_q
            micro_div = 0.0
            if denom > 0:
                micro_price = (best_bid_p * best_ask_q + best_ask_p * best_bid_q) / denom
                micro_div = micro_price - mid_price

            # deltas (vs precedente)
            d_obi = (obi - prev_state['obi']) if prev_state is not None else 0.0
            d_spread = (spread - prev_state['spread']) if prev_state is not None else 0.0

            # aggiorna prev
            prev_state = {'obi': obi, 'spread': spread}
            prev_book_snap = {'bid_p': best_bid_p, 'bid_q': best_bid_q, 'ask_p': best_ask_p, 'ask_q': best_ask_q}

            # --- Save logic + balanced sampling ---
            has_activity = (tick_vol > VOL_FILTER) or (ice_bid > 0.1) or (ice_ask > 0.1) or (abs(d_obi) > 0.2)
            save_datapoint = False
            if has_activity:
                save_datapoint = True
            else:
                # campiona casualmente alcuni datapoint "calmi" per equilibrare distribution
                if np.random.rand() < SAMPLE_RATE_CALM:
                    save_datapoint = True

            if save_datapoint:
                feats = [
                    round(obi, 4),
                    round(d_obi, 4),
                    round(np.log10(tick_vol + 1), 4),
                    round(rolling_cvd_norm, 4),
                    round(ice_bid, 4),
                    round(ice_ask, 4),
                    round(spread, 6),
                    round(d_spread, 6),
                    round(micro_div, 6)
                ]
                # salva il prezzo e il tempo per labeling
                buffer.append({'f': feats, 'p': best_bid_p, 't': now})
                # logging sintetico
                trend = "↗️" if d_obi > 0.05 else ("↘️" if d_obi < -0.05 else "➡️")
                if has_activity:
                    print(f"ACTIVITY OBI:{obi:.3f} {trend} | CVD3s:{rolling_net_vol/1000:.2f}k | IceB:{ice_bid:.3f}", end='\r')
                else:
                    print(f"SAMPLED calm point @ {best_bid_p:.6f}", end='\r')

            # --- 4) Target labeling (lookahead) ---
            # itero su copia per non modificare mentre itero
            now = time.time()
            for item in list(buffer):
                if now - item['t'] >= LOOK_AHEAD:
                    is_profit = 1 if best_bid_p > (item['p'] * (1.0 + TARGET_PROFIT)) else 0
                    row = item['f'] + [is_profit]
                    append_row_csv(FILE_NAME, row, cols)
                    buffer.remove(item)

            # piccolo sleep per evitare busy loop; mantiene ~0.5s ritmo senza dipendere dal sleep per sincronizzazione
            elapsed = time.time() - loop_start
            target_period = 0.5
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    except KeyboardInterrupt:
        print("\n🛑 Miner interrotto dall'utente.")

if __name__ == "__main__":
    main_miner_v7()
