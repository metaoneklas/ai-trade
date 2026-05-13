import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
from typing import Tuple, Dict, Any, Optional

class TradingEnv(gym.Env):
    """
    High-speed Gymnasium environment for Binance RL Trading.
    Expects pre-processed stationary data.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, 
        df: pl.DataFrame, 
        initial_balance: float = 1000.0,
        fee: float = 0.001  # Adjust to 0.0002 if simulating limit orders
    ):
        super(TradingEnv, self).__init__()

        self.initial_balance = initial_balance
        self.fee = fee
        self.max_steps = len(df) - 1

        # Pre-load everything into fast 1D NumPy arrays
        self.prices = df["price"].to_numpy().astype(np.float32)
        self.log_returns = df["log_return"].to_numpy().astype(np.float32)
        self.spreads = df["rel_spread"].to_numpy().astype(np.float32)
        self.ob_imbalances = df["ob_imbalance"].to_numpy().astype(np.float32)
        self.trade_imbalances = df["trade_imbalance"].to_numpy().astype(np.float32)

        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Obs Space (7): LogReturn, Spread, OB Imbalance, Trade Imbalance, NormBal, NormPos, NormEntry
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_quantity = 0.0
        self.entry_price = 0.0
        self.total_fees = 0.0
        
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        current_p = self.prices[self.current_step]
        norm_entry = (self.entry_price / current_p) - 1 if self.entry_price > 0 else 0.0
        
        return np.array([
            self.log_returns[self.current_step],
            self.spreads[self.current_step],
            self.ob_imbalances[self.current_step],
            self.trade_imbalances[self.current_step],
            self.balance / self.initial_balance,
            (self.btc_quantity * current_p) / self.initial_balance,
            norm_entry
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        current_p = self.prices[self.current_step]
        return {
            "step": self.current_step,
            "net_worth": self.balance + (self.btc_quantity * current_p),
            "fees_paid": self.total_fees,
            "btc_quantity": self.btc_quantity,  # <-- ADDED
            "balance": self.balance             # <-- ADDED
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_p = self.prices[self.current_step]
        initial_net_worth = self.balance + (self.btc_quantity * current_p)
        
        # Execute Action
        if action == 1 and self.balance > 0:  # Buy
            cost = self.balance
            fee_amount = cost * self.fee
            self.total_fees += fee_amount
            self.btc_quantity += (cost - fee_amount) / current_p
            self.balance = 0.0
            self.entry_price = current_p
            
        elif action == 2 and self.btc_quantity > 0:  # Sell
            revenue = self.btc_quantity * current_p
            fee_amount = revenue * self.fee
            self.total_fees += fee_amount
            self.balance += (revenue - fee_amount)
            self.btc_quantity = 0.0
            self.entry_price = 0.0

        # Move forward
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        # Calculate Reward
        new_p = self.prices[self.current_step]
        current_net_worth = self.balance + (self.btc_quantity * new_p)
        
        # Logarithmic return of portfolio value, scaled up for stronger RL gradients
        reward = np.log(current_net_worth / initial_net_worth) * 100 
        
        return self._get_obs(), float(reward), terminated, False, self._get_info()