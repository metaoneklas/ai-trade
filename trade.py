import ccxt
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import logging
import os
from collections import deque
from datetime import datetime

# ================= CONFIGURAZIONE UTENTE =================
API_KEY = os.getenv("BINGX_API_KEY")
SECRET_KEY = os.getenv("BINGX_SECRET_KEY")

SYMBOL = "BNB/USDT:USDT" 
TRADE_AMOUNT_USDT = 15.0       
MAX_OPEN_POSITIONS = 1         
ENTRY_THRESHOLD_BPS = 2.5      
TIME_STOP_SECONDS = 20         # Se scatta questo, chiude a mercato e cancella TP/SL

TAKE_PROFIT_BPS = 8.0          # TP Reale sull'exchange
STOP_LOSS_BPS = 4.0            # SL Reale sull'exchange
POST_ONLY = True               

MODEL_FILE = 'xgboost_v15.json' 
ROLLING_WINDOW = 50             
# =========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class BingXHedgeSniper:
    def __init__(self):
        self.exchange = ccxt.bingx({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True
            }
        })
        
        logging.info("⏳ Caricamento mercati exchange...")
        try:
            self.exchange.load_markets()
            logging.info("✅ Mercati caricati.")
        except Exception as e:
            logging.error(f"❌ Errore caricamento mercati: {e}")
            raise e

        # Carica Modello
        if not os.path.exists(MODEL_FILE):
            logging.warning(f"⚠️ Modello {MODEL_FILE} non trovato. Solo test API possibili.")
            self.model = None
        else:
            self.model = xgb.Booster()
            self.model.load_model(MODEL_FILE)
            logging.info(f"🤖 Modello caricato: {MODEL_FILE}")
        
        self.active_trade = None 
        self.history = deque(maxlen=ROLLING_WINDOW)
        self.prev_book = None 
        
        try:
            self.exchange.set_leverage(5, SYMBOL)
        except: pass

    def fetch_market_data(self):
        try:
            book = self.exchange.fetch_order_book(SYMBOL, limit=5)
            bids = book['bids']; asks = book['asks']
            if not bids or not asks: return None

            best_bid_p = bids[0][0]; best_ask_p = asks[0][0]
            mid_price = (best_bid_p + best_ask_p) / 2.0
            
            # Feature Calc (OFI, OBI, etc - Identico al V16)
            spread = best_ask_p - best_bid_p
            sum_bid = sum([x[1] for x in bids]); sum_ask = sum([x[1] for x in asks])
            obi = (sum_bid - sum_ask) / (sum_bid + sum_ask) if (sum_bid+sum_ask) > 0 else 0
            
            # OFI Simplificato per velocità
            ofi = 0.0
            if self.prev_book:
                if best_bid_p > self.prev_book['bid_p']: e_b = bids[0][1]
                elif best_bid_p < self.prev_book['bid_p']: e_b = -self.prev_book['bid_q']
                else: e_b = bids[0][1] - self.prev_book['bid_q']
                
                if best_ask_p < self.prev_book['ask_p']: e_a = asks[0][1]
                elif best_ask_p > self.prev_book['ask_p']: e_a = -self.prev_book['ask_q']
                else: e_a = asks[0][1] - self.prev_book['ask_q']
                ofi = e_b - e_a

            self.prev_book = {'bid_p': best_bid_p, 'bid_q': bids[0][1], 'ask_p': best_ask_p, 'ask_q': asks[0][1]}
            vol_proxy = np.log10(abs(ofi) + 1)
            
            # MicroPrice Div (Feature importante)
            denom = bids[0][1] + asks[0][1]
            mp = (best_bid_p * asks[0][1] + best_ask_p * bids[0][1]) / denom if denom > 0 else mid_price
            micro_div = mp - mid_price

            self.history.append({'obi': obi, 'ofi': ofi, 'vol': vol_proxy, 'spread': spread, 'micro_div': micro_div})
            
            if len(self.history) < 20: return None
            
            df_hist = pd.DataFrame(self.history)
            mean = df_hist.mean(); std = df_hist.std()
            
            features = pd.DataFrame([{
                'z_obi': (obi - mean['obi']) / (std['obi'] + 1e-9),
                'z_ofi': (ofi - mean['ofi']) / (std['ofi'] + 1e-9),
                'z_vol': (vol_proxy - mean['vol']) / (std['vol'] + 1e-9),
                'z_spread': (spread - mean['spread']) / (std['spread'] + 1e-9),
                'z_micro_div': (micro_div - mean['micro_div']) / (std['micro_div'] + 1e-9)
            }])
            return features, mid_price, best_bid_p, best_ask_p

        except Exception as e:
            logging.error(f"Errore dati: {e}"); return None

    def execute_entry(self, direction, price):
        """Piazza solo l'ordine di ingresso Limit"""
        try:
            amount = self.exchange.amount_to_precision(SYMBOL, TRADE_AMOUNT_USDT / price)
            params = {'positionSide': direction}
            if POST_ONLY: params['postOnly'] = True
            side = 'buy' if direction == 'LONG' else 'sell'
            
            logging.info(f"🔫 ENTRY: {direction} {amount} @ {price}")
            order = self.exchange.create_order(SYMBOL, 'limit', side, amount, price, params)
            return order, direction, amount
        except ccxt.OrderImmediatelyFillable:
            logging.warning("⚠️ Ordine Maker rifiutato (sarebbe Taker). Skip."); return None, None, None
        except Exception as e:
            logging.error(f"Errore ordine entry: {e}"); return None, None, None

    def place_hard_exits(self, direction, amount, entry_price):
        """
        Piazza ordini REALI di TP e SL sull'exchange.
        """
        try:
            # Calcolo Prezzi
            if direction == 'LONG':
                tp_price = entry_price * (1 + TAKE_PROFIT_BPS / 10000)
                sl_price = entry_price * (1 - STOP_LOSS_BPS / 10000)
                exit_side = 'sell'
            else: # SHORT
                tp_price = entry_price * (1 - TAKE_PROFIT_BPS / 10000)
                sl_price = entry_price * (1 + STOP_LOSS_BPS / 10000)
                exit_side = 'buy'
            
            # Formatta prezzi
            tp_price = self.exchange.price_to_precision(SYMBOL, tp_price)
            sl_price = self.exchange.price_to_precision(SYMBOL, sl_price)
            
            logging.info(f"🛡️ Piazzamento Hard Stops: TP={tp_price}, SL={sl_price}")

            # TP Order (TAKE_PROFIT_MARKET)
            tp_params = {'positionSide': direction, 'stopPrice': tp_price}
            tp_order = self.exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', exit_side, amount, params=tp_params)
            
            # SL Order (STOP_MARKET)
            sl_params = {'positionSide': direction, 'stopPrice': sl_price}
            sl_order = self.exchange.create_order(SYMBOL, 'STOP_MARKET', exit_side, amount, params=sl_params)
            
            return [tp_order['id'], sl_order['id']]

        except Exception as e:
            logging.error(f"❌ ERRORE PIAZZAMENTO TP/SL: {e}")
            logging.warning("⚠️ Chiudo posizione immediatamente per sicurezza!")
            self.close_position_market("Errore TP/SL", direction, amount)
            return []

    def close_position_market(self, reason, direction=None, amount=None):
        """Chiude a mercato e cancella ordini pendenti"""
        try:
            # 1. Cancella TP/SL pendenti per evitare doppi eseguiti
            self.exchange.cancel_all_orders(SYMBOL)
            
            # 2. Chiudi Posizione (se non passati come arg, prova a recuperarli)
            if not direction or not amount:
                if self.active_trade:
                    direction = self.active_trade['direction']
                    amount = self.active_trade['amount']
                else: return

            side = 'sell' if direction == 'LONG' else 'buy'
            params = {'positionSide': direction}
            
            self.exchange.create_order(SYMBOL, 'market', side, amount, params=params)
            logging.info(f"🏁 Posizione chiusa ({reason}) e ordini cancellati.")
            self.active_trade = None
            
        except Exception as e:
            logging.error(f"Errore Panic Close: {e}")

    def run(self):
        if not self.model: return
        logging.info("🚀 Bot Sniper (Hard Stops) avviato...")
        
        while True:
            try:
                # 1. Check TP/SL Execution (Polling stato ordine)
                # Se un TP/SL viene fillato, active_trade deve essere resettato.
                if self.active_trade:
                    # Invece di controllare il prezzo locale, controlliamo se abbiamo ancora una posizione aperta
                    try:
                        positions = self.exchange.fetch_positions([SYMBOL])
                        my_pos = [p for p in positions if float(p['contracts']) > 0]
                        if not my_pos:
                            logging.info("✨ Posizione chiusa dall'exchange (TP o SL colpito).")
                            self.exchange.cancel_all_orders(SYMBOL) # Pulizia extra
                            self.active_trade = None
                            time.sleep(1)
                            continue
                    except: pass # Errore rete temporaneo

                    # Time Stop Check
                    elapsed = time.time() - self.active_trade['entry_time']
                    if elapsed >= TIME_STOP_SECONDS:
                        self.close_position_market("Time Stop", self.active_trade['direction'], self.active_trade['amount'])
                        time.sleep(1)
                        continue

                # 2. Market Data & Prediction
                data = self.fetch_market_data()
                if not data: time.sleep(0.5); continue
                features, mid, bid, ask = data
                
                # Se abbiamo un trade attivo, saltiamo predizione (abbiamo già gli Hard Stops)
                if self.active_trade:
                    time.sleep(0.5)
                    continue

                # 3. Entry Logic
                dmatrix = xgb.DMatrix(features)
                pred_bps = self.model.predict(dmatrix)[0]
                
                if abs(pred_bps) > ENTRY_THRESHOLD_BPS:
                    direction = 'LONG' if pred_bps > 0 else 'SHORT'
                    price = bid if direction == 'LONG' else ask 
                    
                    logging.info(f"Signal: {pred_bps:.2f} bps. Tentativo {direction}...")
                    order, dir_confirmed, amt = self.execute_entry(direction, price)
                    
                    if order:
                        time.sleep(1) # Attesa Fill
                        try:
                            status = self.exchange.fetch_order(order['id'], SYMBOL)
                            if status['status'] == 'filled':
                                avg_price = float(status['average']) if status['average'] else price
                                logging.info(f"✅ ENTRY FILLATO! {dir_confirmed} @ {avg_price}")
                                
                                # --- PIAZZAMENTO HARD STOPS ORA ---
                                exit_ids = self.place_hard_exits(dir_confirmed, amt, avg_price)
                                
                                self.active_trade = {
                                    'id': order['id'],
                                    'direction': dir_confirmed,
                                    'price': avg_price,
                                    'amount': amt,
                                    'entry_time': time.time(),
                                    'exit_order_ids': exit_ids
                                }
                            else:
                                self.exchange.cancel_order(order['id'], SYMBOL)
                                logging.info("❌ Entry non fillato. Cancellato.")
                        except Exception as e:
                            logging.error(f"Errore gestione fill: {e}")

                time.sleep(0.5)

            except Exception as e:
                logging.error(f"Loop error: {e}"); time.sleep(5)

    def test_api_execution(self):
        logging.info("🧪 AVVIO TEST API COMPLETO (Ordini + TP/SL)...")
        try:
            # 1. Prep
            market = self.exchange.market(SYMBOL)
            min_amount = market['limits']['amount']['min']
            ticker = self.exchange.fetch_ticker(SYMBOL)
            price = ticker['last']
            target_usdt = max(min_amount * price * 1.5, 11.0)
            amount = self.exchange.amount_to_precision(SYMBOL, target_usdt / price)
            
            # 2. Apertura LONG
            logging.info(f"1️⃣  Apro LONG Test: {amount} {SYMBOL}...")
            order = self.exchange.create_order(SYMBOL, 'market', 'buy', amount, params={'positionSide': 'LONG'})
            logging.info("✅ Long Aperto.")
            time.sleep(2)
            
            # Recupera prezzo medio ingresso
            pos_info = self.exchange.fetch_position(SYMBOL) # Potrebbe richiedere fetch_positions con filtro
            # Fallback generico
            entry_price = price
            
            # 3. Imposto TP e SL reali (Ordini Condizionali)
            tp_price = entry_price * 1.05 # +5%
            sl_price = entry_price * 0.95 # -5%
            
            logging.info(f"2️⃣  Imposto TP (@{tp_price:.2f}) e SL (@{sl_price:.2f})...")
            
            # TP Order (TAKE_PROFIT_MARKET)
            # In Hedge Mode: Sell Long
            tp_params = {'positionSide': 'LONG', 'stopPrice': tp_price}
            tp_order = self.exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', 'sell', amount, params=tp_params)
            logging.info(f"✅ TP Piazzato (ID: {tp_order['id']})")
            
            # SL Order (STOP_MARKET)
            sl_params = {'positionSide': 'LONG', 'stopPrice': sl_price}
            sl_order = self.exchange.create_order(SYMBOL, 'STOP_MARKET', 'sell', amount, params=sl_params)
            logging.info(f"✅ SL Piazzato (ID: {sl_order['id']})")
            
            # 4. Verifica Ordini Aperti
            open_orders = self.exchange.fetch_open_orders(SYMBOL)
            logging.info(f"📋 Ordini Aperti Rilevati: {len(open_orders)}")
            for o in open_orders:
                logging.info(f"   - {o['type']} {o['side']} @ {o.get('stopPrice') or o.get('price')}")

            # 5. Attesa
            logging.info("⏳ Attesa 5 secondi per verifica stabilità...")
            time.sleep(30)
            
            # 6. Cancellazione TP/SL
            logging.info("🗑️  Cancello ordini TP/SL...")
            self.exchange.cancel_all_orders(SYMBOL)
            logging.info("✅ Ordini cancellati.")
            
            # 7. Chiusura Posizione
            logging.info("🏁 Chiudo Posizione a mercato...")
            self.exchange.create_order(SYMBOL, 'market', 'sell', amount, params={'positionSide': 'LONG'})
            
            # Verifica finale
            time.sleep(1)
            positions = self.exchange.fetch_positions([SYMBOL])
            active = [p for p in positions if float(p['contracts']) > 0]
            if not active:
                logging.info("✅ TEST SUPERATO: Ciclo completo (Open -> TP/SL -> Cancel -> Close) OK.")
            else:
                logging.error("❌ ATTENZIONE: La posizione sembra ancora aperta!")

        except Exception as e:
            logging.error(f"❌ ERRORE TEST: {e}")
            # Tenta chiusura emergenza
            try:
                logging.warning("⚠️ Tento chiusura emergenza...")
                self.exchange.cancel_all_orders(SYMBOL)
                self.exchange.create_order(SYMBOL, 'market', 'sell', amount, params={'positionSide': 'LONG'})
            except: pass


if __name__ == "__main__":
    bot = BingXHedgeSniper()
    # bot.test_api_execution() # Decommenta se vuoi testare
    bot.run()