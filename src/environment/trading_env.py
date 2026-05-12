import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
from typing import Tuple, Dict, Any, Optional

class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for cryptocurrency trading based on 
    Binance AggTrades and Depth10 Orderbook data.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, 
        df: pl.DataFrame, 
        initial_balance: float = 1000.0,
        fee: float = 0.001,  # 0.1% fee
        max_steps: Optional[int] = None
    ):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.fee = fee
        self.max_steps = max_steps if max_steps else len(df) - 1

        # Action Space: 0 = Hold, 1 = Buy (Market), 2 = Sell (Market)
        self.action_space = spaces.Discrete(3)

        # Observation Space:
        # - Price: 1
        # - Quantity: 1
        # - Bids (10 prices + 10 quantities): 20
        # - Asks (10 prices + 10 quantities): 20
        # - Balance (USDT): 1
        # - Position (BTC): 1
        # - Entry Price: 1
        # Total: 1 + 1 + 20 + 20 + 1 + 1 + 1 = 45 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.btc_quantity = 0.0
        self.entry_price = 0.0
        self.total_fees = 0.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def _get_observation(self) -> np.ndarray:
        row = self.df.row(self.current_step, named=True)
        
        # Extract features
        price = row["price"]
        quantity = row["quantity"]
        bids = row["bids_vec"]
        asks = row["asks_vec"]
        
        obs = np.array([
            price,
            quantity,
            *bids,
            *asks,
            self.balance,
            self.btc_quantity,
            self.entry_price
        ], dtype=np.float32)
        
        return obs

    def _get_info(self) -> Dict[str, Any]:
        current_price = self.df.row(self.current_step, named=True)["price"]
        net_worth = self.balance + (self.btc_quantity * current_price)
        return {
            "step": self.current_step,
            "balance": self.balance,
            "btc_quantity": self.btc_quantity,
            "net_worth": net_worth,
            "total_fees": self.total_fees
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Get current price data
        row = self.df.row(self.current_step, named=True)
        current_price = row["price"]
        
        # Portfolio value before action
        initial_net_worth = self.balance + (self.btc_quantity * current_price)
        
        # Execute Action
        if action == 1:  # Buy
            if self.balance > 0:
                cost = self.balance
                fee_amount = cost * self.fee
                self.total_fees += fee_amount
                
                # Simple slippage: use the best ask if available (simplified for now)
                # In a more complex model, we'd walk the orderbook.
                execution_price = current_price * 1.0001 # Assume 0.01% slippage for market buy
                
                self.btc_quantity += (cost - fee_amount) / execution_price
                self.balance = 0.0
                self.entry_price = execution_price
                
        elif action == 2:  # Sell
            if self.btc_quantity > 0:
                revenue = self.btc_quantity * current_price
                fee_amount = revenue * self.fee
                self.total_fees += fee_amount
                
                execution_price = current_price * 0.9999 # Assume 0.01% slippage for market sell
                
                self.balance += (revenue - fee_amount)
                self.btc_quantity = 0.0
                self.entry_price = 0.0

        # Move to next step
        self.current_step += 1
        
        # Termination conditions
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Calculate Reward
        new_row = self.df.row(self.current_step, named=True)
        new_price = new_row["price"]
        current_net_worth = self.balance + (self.btc_quantity * new_price)
        
        # Reward is the change in net worth (percentage or absolute)
        reward = (current_net_worth - initial_net_worth) / initial_net_worth
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

if __name__ == "__main__":
    # Mock data for testing
    import polars as pl
    import numpy as np
    
    data = {
        "price": np.random.uniform(50000, 60000, 100),
        "quantity": np.random.uniform(0.1, 1.0, 100),
        "bids_vec": [list(np.random.uniform(50000, 60000, 20)) for _ in range(100)],
        "asks_vec": [list(np.random.uniform(50000, 60000, 20)) for _ in range(100)],
    }
    df = pl.DataFrame(data)
    
    env = TradingEnv(df)
    obs, info = env.reset()
    print(f"Initial Obs: {obs.shape}")
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.6f}, Net Worth: {info['net_worth']:.2f}")
