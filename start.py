import random
import numpy as np
from stable_baselines3 import PPO
from learn import main as init_env
from typing import Optional

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"[*] Seed set to {seed}")

def run_evaluation(model_path: Optional[str] = None, num_episodes: int = 1):
    """
    Runs evaluation episodes using a trained model or random actions.
    """
    set_seed(42)

    # 1. Initialize Environment
    # We reuse the initialization logic from learn.py
    env, _ = init_env()
    if env is None:
        print("[!] Environment could not be initialized. Check data availability.")
        return

    # 2. Load Model
    model = None
    if model_path:
        print(f"[*] Loading model from {model_path}...")
        try:
            model = PPO.load(model_path, env=env)
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            print("[*] Proceeding with random actions for testing.")
    else:
        print("[*] No model path provided. Proceeding with random actions for testing.")

    # 3. Evaluation Loop
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        
        print(f"\n--- Episode {episode + 1} Start ---")
        
        while not (done or truncated):
            if model:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Optional: Print progress every 1000 steps
            if info['step'] % 1000 == 0:
                print(f"Step: {info['step']}, Net Worth: {info['net_worth']:.2f}, Balance: {info['balance']:.2f}, BTC: {info['btc_quantity']:.4f}")

        print(f"\n--- Episode {episode + 1} Finished ---")
        print(f"Total Reward: {total_reward:.6f}")
        print(f"Final Net Worth: {info['net_worth']:.2f}")
        print(f"Total Fees Paid: {info['total_fees']:.2f}")
        print(f"Profit/Loss: {info['net_worth'] - 1000.0:.2f}") # Assuming 1000 initial balance

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start the RL Trading Bot Evaluation")
    parser.add_argument("--model", type=str, help="Path to the trained model file", default=None)
    parser.add_argument("--episodes", type=int, help="Number of evaluation episodes", default=1)
    
    args = parser.parse_args()
    
    run_evaluation(model_path=args.model, num_episodes=args.episodes)
