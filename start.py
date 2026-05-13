import os
import argparse
import numpy as np
import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environment.trading_env import TradingEnv  # Assuming your environment is in env.py

def evaluate_model(model_path: str, stats_path: str, data_path: str):
    print(f"[*] Loading data from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[!] Cannot find {data_path}. Run preprocess.py first.")
    
    df = pl.read_parquet(data_path)
    
    # 1. Use ONLY the Validation Data (Last 20%)
    train_size = int(len(df) * 0.8)
    val_df = df.slice(train_size, len(df) - train_size)
    print(f"[*] Evaluating on unseen validation set: {len(val_df)} records.")

    # 2. Setup Environment
    env_maker = lambda: TradingEnv(val_df)
    vec_env = DummyVecEnv([env_maker])

    # 3. Load Normalization Stats
    print(f"[*] Loading Normalization Stats from {stats_path}...")
    if os.path.exists(stats_path):
        vec_env = VecNormalize.load(stats_path, vec_env)
        # CRITICAL: Do not update stats during testing, and do not normalize rewards
        vec_env.training = False 
        vec_env.norm_reward = False
    else:
        print("[!] WARNING: No VecNormalize stats found. Model will likely fail.")

    # 4. Load Model
    print(f"[*] Loading PPO Model from {model_path}...")
    try:
        model = PPO.load(model_path, env=vec_env)
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        return

    # 5. Evaluation Variables
    # Note: SB3 VecEnv reset() returns just the observation array
    obs = vec_env.reset()
    done = False
    
    # Metrics
    peak_net_worth = 1000.0
    max_drawdown = 0.0
    buy_count = 0
    sell_count = 0

    print("\n[*] Starting Simulation...")
    
    while not done:
        # Predict the action (deterministic=True is crucial for evaluation)
        action, _states = model.predict(obs, deterministic=True)
        
        # Note: SB3 VecEnv step() returns 4 values, and they are batched
        obs, rewards, dones, infos = vec_env.step(action)
        
        # Extract info from the first (and only) environment
        info = infos[0]
        act = action[0]
        
        # Track Actions
        if act == 1:
            buy_count += 1
        elif act == 2:
            sell_count += 1

        # Track Drawdown
        current_nw = info['net_worth']
        if current_nw > peak_net_worth:
            peak_net_worth = current_nw
        
        drawdown = (peak_net_worth - current_nw) / peak_net_worth
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # Print progress every 10,000 steps
        step = info['step']
        if step % 10000 == 0:
            print(f"Step {step} | Net Worth: ${current_nw:.2f} | Action: {act}")

        done = dones[0]

    # 6. Final Report
    final_net_worth = info['net_worth']
    profit_loss = final_net_worth - 1000.0
    roi = (profit_loss / 1000.0) * 100

    print("\n" + "="*40)
    print("      EVALUATION RESULTS")
    print("="*40)
    print(f"Initial Balance  : $1000.00")
    print(f"Final Net Worth  : ${final_net_worth:.2f}")
    print(f"Total Profit/Loss: ${profit_loss:.2f} ({roi:.2f}%)")
    print(f"Total Fees Paid  : ${info['fees_paid']:.2f}")
    print(f"Max Drawdown     : {max_drawdown * 100:.2f}%")
    print("-" * 40)
    print(f"Total Buy Orders : {buy_count}")
    print(f"Total Sell Orders: {sell_count}")
    print(f"Total Trades     : {buy_count + sell_count}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Trading Bot")
    # Default paths based on previous scripts
    parser.add_argument("--model", type=str, default="models/ppo_trading_final.zip")
    parser.add_argument("--stats", type=str, default="models/vec_normalize.pkl")
    parser.add_argument("--data", type=str, default="data/processed_rl_data.parquet")
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.stats, args.data)