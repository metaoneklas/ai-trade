import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# ================= CONFIGURAZIONE =================
DATA_FILE = 'trade.csv'
MODEL_FILE = 'xgboost_v15.json'

# --- PARAMETRI ---
# Hai detto 0.05. Se intendi 0.05%, scrivi 5.0. Se intendi 0.05 bps, lascia 0.05.
TRADING_FEE_BPS = 5   

# SOGLIA
PERCENTILE_THRESHOLD = 90 # Entra sul top 10% dei segnali

# IMPORTANTE: Se True, trada anche se il profitto previsto non copre le fee
FORCE_TRADES = True 
# ==============================================

def run_diagnostic():
    print("--- 🩺 DIAGNOSTIC BACKTEST (FORCE TRADES) ---")
    
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_FILE):
        return

    # 1. Carica
    df = pd.read_csv(DATA_FILE).dropna()
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()
    X_test = df_test.drop(columns=['target_return_bps', 'raw_price', 'mid_price', 'p_entry'], errors='ignore')
    
    # 2. Predizioni
    print(f"🔮 Generazione predizioni...")
    df_test['pred_bps'] = model.predict(X_test)
    
    # 3. Analisi Volatilità Predetta
    avg_pred = np.mean(np.abs(df_test['pred_bps']))
    p90_pred = np.percentile(np.abs(df_test['pred_bps']), 90)
    print(f"📊 Volatilità Modello (Media): {avg_pred:.4f} bps")
    print(f"📊 Volatilità Modello (Top 10%): {p90_pred:.4f} bps")
    
    # 4. Calcolo Soglia
    threshold = p90_pred
    
    if not FORCE_TRADES:
        # Vecchia logica break-even
        be_thresh = (TRADING_FEE_BPS + 0.1) 
        threshold = max(threshold, be_thresh)
        print(f"🛡️ Break-Even attivo. Soglia minima: {be_thresh:.4f}")
    else:
        print(f"🔥 FORCE TRADES ATTIVO. Ignoro break-even.")

    print(f"👉 SOGLIA USATA: {threshold:.4f} bps")

    # 5. Logica Segnali
    df_test['signal'] = 0
    df_test.loc[df_test['pred_bps'] > threshold, 'signal'] = 1
    df_test.loc[df_test['pred_bps'] < -threshold, 'signal'] = -1
    
    # PnL Lordo (Senza Fee)
    df_test['gross_pnl_bps'] = df_test['signal'] * df_test['target_return_bps']
    
    # PnL Netto (Con Fee)
    df_test['net_pnl_bps'] = df_test['gross_pnl_bps'] - (abs(df_test['signal']) * TRADING_FEE_BPS)
    
    df_test['equity_gross'] = df_test['gross_pnl_bps'].cumsum()
    df_test['equity_net'] = df_test['net_pnl_bps'].cumsum()

    # 6. Risultati
    n_trades = df_test[df_test['signal'] != 0].shape[0]
    
    if n_trades == 0:
        print("❌ Ancora 0 trade. La soglia è troppo alta per i dati attuali.")
        return

    win_rate_gross = len(df_test[df_test['gross_pnl_bps'] > 0]) / n_trades
    total_net = df_test['net_pnl_bps'].sum()
    total_gross = df_test['gross_pnl_bps'].sum()
    
    print(f"\n📊 RISULTATI (Fee = {TRADING_FEE_BPS} bps)")
    print(f"   ---------------------------")
    print(f"   Trade Totali:    {n_trades}")
    print(f"   Win Rate (Puri): {win_rate_gross:.2%}")
    print(f"   PnL LORDO:       {total_gross:.2f} bps (Bravura Modello)")
    print(f"   PnL NETTO:       {total_net:.2f} bps (Portafoglio Reale)")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_test['equity_gross'].values, label='Equity LORDA (No Fee)', color='blue', alpha=0.6)
    plt.plot(df_test['equity_net'].values, label=f'Equity NETTA (Fee {TRADING_FEE_BPS})', color='green', linewidth=2)
    plt.title(f"Diagnostic Backtest (Thresh: {threshold:.4f} bps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('diagnostic_result.png')
    print("\n🖼️ Grafico: diagnostic_result.png")

if __name__ == "__main__":
    run_diagnostic()