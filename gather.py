 import asyncio

import websockets

import json

import pandas as pd

import os

from datetime import datetime


# Configurazioni

SYMBOL = "btcusdt"

WS_URL = f"wss://stream.binance.com:9443/stream?streams={SYMBOL}@aggTrade/{SYMBOL}@depth10@100ms"


# Quanti record accumulare in RAM prima di salvare su disco

BATCH_SIZE = 5000


# Cartelle di output

TRADE_DIR = "data/trades"

OB_DIR = "data/orderbook"


def init_directories():

    """Crea le cartelle per i file Parquet se non esistono."""

    os.makedirs(TRADE_DIR, exist_ok=True)

    os.makedirs(OB_DIR, exist_ok=True)


def save_to_parquet(data_buffer, folder, prefix):

    """Converte il buffer in DataFrame e lo salva in Parquet."""

    if not data_buffer:

        return

    

    df = pd.DataFrame(data_buffer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{folder}/{prefix}_{timestamp}.parquet"

    

    # pyarrow è l'engine di default di pandas per parquet, molto efficiente

    df.to_parquet(filename, index=False, engine='pyarrow')

    print(f"[+] Salvato batch di {len(data_buffer)} record in {filename}")

    

    # Svuota il buffer in modo sicuro

    data_buffer.clear()


async def data_collector():

    init_directories()

    

    trades_buffer = []

    ob_buffer = []

    

    print(f"[*] Connessione al WebSocket per {SYMBOL.upper()}...")

    

    # Assicuriamoci che i dati vengano salvati anche se fermiamo lo script con Ctrl+C

    try:

        async for ws in websockets.connect(WS_URL):

            try:

                print("[*] Connesso! In attesa di dati. I file Parquet verranno creati a blocchi.")

                while True:

                    message = await ws.recv()

                    data = json.loads(message)

                    

                    stream = data.get('stream', '')

                    payload = data.get('data', {})

                    current_time = datetime.now().isoformat()


                    # Gestione Stream Trades Aggregati

                    if 'aggTrade' in stream:

                        trades_buffer.append({

                            'timestamp': current_time,

                            'price': float(payload['p']),

                            'quantity': float(payload['q']),

                            'is_buyer_maker': bool(payload['m'])

                        })

                        

                        if len(trades_buffer) >= BATCH_SIZE:

                            save_to_parquet(trades_buffer, TRADE_DIR, "trades")

                            

                    # Gestione Stream Orderbook

                    elif 'depth' in stream:

                        ob_buffer.append({

                            'timestamp': current_time,

                            'bids': json.dumps(payload['bids']),

                            'asks': json.dumps(payload['asks'])

                        })

                        

                        if len(ob_buffer) >= BATCH_SIZE:

                            save_to_parquet(ob_buffer, OB_DIR, "orderbook")


            except websockets.ConnectionClosed:

                print("[!] Connessione persa. Riconnessione in corso...")

                continue

            except Exception as e:

                print(f"[!] Errore inaspettato durante la ricezione: {e}")

                break


    finally:

        # Salvataggio di emergenza: se lo script si ferma, salva i record rimasti nei buffer

        print("\n[*] Chiusura... Salvataggio dei dati rimanenti nei buffer.")

        save_to_parquet(trades_buffer, TRADE_DIR, "trades_final")

        save_to_parquet(ob_buffer, OB_DIR, "orderbook_final")


if __name__ == "__main__":

    try:

        asyncio.run(data_collector())

    except KeyboardInterrupt:

        print("\n[*] Raccolta dati interrotta dall'utente.") 