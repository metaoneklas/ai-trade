import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from src.environment.trading_env import TradingEnv

import os

def main():
    data_path = "data/processed_rl_data.parquet"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}. Run preprocess.py first.")

    print("[*] Loading preprocessed data...")
    df = pl.read_parquet(data_path)
    
    # 1. Temporal Split (80/20) - DO NOT SHUFFLE TRADING DATA
    train_size = int(len(df) * 0.8)
    train_df = df.slice(0, train_size)
    val_df = df.slice(train_size, len(df) - train_size)
    print(f"[*] Train set: {len(train_df)} rows. Validation set: {len(val_df)} rows.")

    # 2. Vectorize and Normalize Environment
    # Using 4 parallel environments speeds up trajectory collection
    env_maker = lambda: TradingEnv(train_df)
    vec_env = DummyVecEnv([env_maker for _ in range(4)])
    
    # CRITICAL: Normalize observations so the Neural Network doesn't blow up
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Define PPO Model for Real-World Noise
    print("[*] Initializing PPO Agent...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        tensorboard_log="./logs/ppo_trading/",
        learning_rate=5e-5,        # Low LR for volatile market data
        n_steps=4096,              # Large batch of steps per update
        batch_size=256,
        n_epochs=5,                # Low epochs to prevent overfitting
        gamma=0.99,                # Future reward discount
        clip_range=0.1,            # Tighter clipping for stability
        ent_coef=0.01,             # Entropy to encourage exploration
        device="auto"              # Uses GPU if available
    )

    # Save a checkpoint every 500,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=125_000, # 125k * 4 envs = 500k timesteps
        save_path='./models/checkpoints/',
        name_prefix='ppo_trading'
    )

    # 4. Train Model
    print("[*] Starting Training Loop...")
    os.makedirs("models", exist_ok=True)
    
    # 2,000,000 timesteps is a good starting point for ~700k train rows
    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback) 
    
    # 5. Save Final Model & Normalization Stats
    model.save("models/ppo_trading_final")
    vec_env.save("models/vec_normalize.pkl")
    print("[+] Training Complete. Model and Normalization stats saved to /models/")

if __name__ == "__main__":
    main()