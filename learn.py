import random
import numpy as np
import polars as pl
from src.data.loader import TradingDataLoader
from src.environment.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from typing import Optional

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed) if torch is used directly
    print(f"[*] Seed set to {seed}")

def main(data_path: Optional[str] = None):
    set_seed(42)

    # 1. Load and Preprocess Data
    print("[*] Loading data...")
    # For now, we assume data exists in data/trades and data/orderbook
    # If not, we'll catch the error and suggest gathering data first.
    try:
        loader = TradingDataLoader()
        df = loader.load_aligned_data()
        
        # Take a subset or full dataset
        # processed_df = loader.preprocess_for_rl(df.head(10000))
        processed_df = loader.preprocess_for_rl(df)
        
        print(f"[*] Data loaded: {len(processed_df)} records.")
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        print("[!] Make sure you have collected data using gather.py and it's stored in data/trades and data/orderbook.")
        return

    # 2. Initialize Environment
    print("[*] Initializing Environment...")
    env = TradingEnv(processed_df)

    # 3. Build Model (Template)
    print("[*] Building RL Model (PPO)...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/ppo_trading/",
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    # 4. Training (Optional/Template)
    # print("[*] Starting Training...")
    # model.learn(total_timesteps=100000)
    # model.save("models/ppo_trading_bot")
    # print("[*] Model saved to models/ppo_trading_bot")

    print("[*] Environment and model ready.")
    return env, model

if __name__ == "__main__":
    main()
