import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import os
import joblib

# ================= CONFIGURAZIONE =================
# Nome del file generato dal Miner V15. 
# Se hai usato un altro nome, modificalo qui.
DATA_FILE = 'trade.csv' 
MODEL_FILE = 'xgboost_v15.json'
TEST_SIZE_PCT = 0.2
# ==============================================

def train_v15():
    print(f"--- 🧠 TRAINING V15 (OFI + ROLLING NORM) ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Errore: File '{DATA_FILE}' non trovato.")
        print("   Assicurati di aver rinominato il file o aggiorna la variabile DATA_FILE nello script.")
        return

    # 1. Caricamento Dati
    print("⏳ Caricamento dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"📊 Righe totali: {len(df)}")
    
    df = df.dropna()

    # 2. Preparazione
    target_col = 'target_return_bps'
    
    # Rimuoviamo colonne che non sono feature predittive
    # 'raw_price' serve solo per debug umano o calcoli post-hoc, non per il modello (non è stazionario)
    # 'p_entry' se presente va tolto
    cols_to_drop = [target_col, 'raw_price', 'mid_price', 'p_entry']
    
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df[target_col]
    
    print(f"🎯 Features ({len(X.columns)}): {X.columns.tolist()}")

    # 3. Split Cronologico (Mai fare shuffle sui time series!)
    split_idx = int(len(df) * (1 - TEST_SIZE_PCT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"✂️ Split: Train={len(X_train)} | Test={len(X_test)}")

    # 4. Configurazione Modello
    # Usiamo parametri conservativi per evitare overfitting su 80k righe
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,       # Abbondiamo, tanto c'è early_stopping
        learning_rate=0.03,      # Lento e preciso
        max_depth=4,             # Alberi poco profondi (evita memorizzazione rumore)
        subsample=0.7,           # Usa il 70% delle righe per albero
        colsample_bytree=0.7,    # Usa il 70% delle colonne per albero
        random_state=42,
        n_jobs=-1,
        # Parametri spostati nel costruttore per compatibilità XGBoost nuovi
        eval_metric='rmse',
        early_stopping_rounds=50
    )

    # 5. Training
    print("🚀 Avvio addestramento...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    # 6. Valutazione
    preds = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"\n📉 Error Metrics:")
    print(f"   RMSE: {rmse:.4f} bps")
    print(f"   MAE:  {mae:.4f} bps")

    # 7. Accuratezza Direzionale (Il test vero)
    # Filtriamo i casi dove il target è 0 (mercato fermo)
    mask = y_test != 0
    if mask.sum() > 0:
        y_sign = np.sign(y_test[mask])
        p_sign = np.sign(preds[mask])
        acc = accuracy_score(y_sign, p_sign)
        print(f"\n🧭 Directional Accuracy (su {mask.sum()} trade attivi):")
        print(f"   ACCURACY: {acc:.2%} " + ("✅ BUONO" if acc > 0.55 else "⚠️ NEUTRO"))
    else:
        print("\n⚠️ Test set troppo statico (tutti zeri).")

    # 8. Feature Importance (Verifica Teoria Microstruttura)
    # Se z_ofi o z_obi sono in alto, il report aveva ragione.
    results = pd.DataFrame({
        'Feature': X.columns,
        'Importanza': model.feature_importances_
    }).sort_values(by='Importanza', ascending=False)
    
    print("\n🏆 Top Features:")
    print(results.head(10))

    # Grafico
    plt.figure(figsize=(10, 6))
    plt.barh(results['Feature'][:10], results['Importanza'][:10])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance (V15)")
    plt.savefig('importance_v15.png')
    print("🖼️ Grafico salvato: importance_v15.png")

    # Salvataggio
    model.save_model(MODEL_FILE)
    print(f"💾 Modello salvato: {MODEL_FILE}")

if __name__ == "__main__":
    train_v15()