import pandas as pd
import matplotlib.pyplot as plt
import os

# Configurazione
FILE_NAME = 'training_data_final.csv'
OUTPUT_IMAGE = 'correlation_analysis.png'

def main_visual():
    if not os.path.exists(FILE_NAME):
        print(f"❌ Errore: Il file '{FILE_NAME}' non esiste. Fai girare il miner prima.")
        return

    print(f"Caricamento dati da {FILE_NAME}...")
    df = pd.read_csv(FILE_NAME)

    # Filtra solo i dati dove c'è stato movimento reale
    # Escludiamo lo zero assoluto che rappresenta spesso mancanza di dati o mercato fermo
    df_active = df[df['target_return_bps'] != 0]

    print(f"Righe totali: {len(df)}")
    print(f"Righe attive (con movimento): {len(df_active)}")

    if len(df_active) < 10:
        print("⚠️ Troppi pochi dati per generare un grafico significativo.")
        return

    plt.figure(figsize=(12, 8))
    
    # Scatter plot: OBI vs Future Return
    # alpha=0.1 rende i punti trasparenti per vedere dove si accumulano (densità)
    plt.scatter(df_active['obi'], df_active['target_return_bps'], alpha=0.1, c='royalblue', s=10)
    
    # Linea orizzontale e verticale a 0 per riferimento
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    plt.title(f"Market Microstructure: OBI vs Future Return (n={len(df_active)})")
    plt.xlabel("Order Book Imbalance (-1: Sell Pressure, +1: Buy Pressure)")
    plt.ylabel("Future Return (Basis Points)")
    plt.grid(True, alpha=0.2)
    
    # SALVATAGGIO SU FILE INVECE DI SHOW()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"✅ Grafico salvato con successo: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main_visual()