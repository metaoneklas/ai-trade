import polars as pl
import json
from pathlib import Path
from typing import List, Optional

class TradingDataLoader:
    """
    Handles loading and alignment of Binance AggTrades and Depth10 Orderbook data.
    Optimized for memory efficiency using Polars.
    """

    def __init__(self, trades_dir: str = "data/trades", ob_dir: str = "data/orderbook"):
        self.trades_dir = Path(trades_dir)
        self.ob_dir = Path(ob_dir)

    def load_aligned_data(self) -> pl.DataFrame:
        """
        Loads trades and orderbook data, parses JSON fields, and joins them on timestamp.
        """
        # Load trades
        trades_files = list(self.trades_dir.glob("*.parquet"))
        if not trades_files:
            raise FileNotFoundError(f"No parquet files found in {self.trades_dir}")
        
        trades_df = pl.read_parquet(trades_files).with_columns([
            pl.col("timestamp").str.to_datetime()
        ]).sort("timestamp")

        # Load orderbook
        ob_files = list(self.ob_dir.glob("*.parquet"))
        if not ob_files:
            raise FileNotFoundError(f"No parquet files found in {self.ob_dir}")
        
        ob_df = pl.read_parquet(ob_files).with_columns([
            pl.col("timestamp").str.to_datetime()
        ]).sort("timestamp")

        # Join data on timestamp using 'join_asof' for nearest match if needed, 
        # but here we assume they might be close. Given 100ms depth updates, 
        # join_asof is safer.
        aligned_df = pl.join_asof(
            trades_df,
            ob_df,
            on="timestamp",
            strategy="backward"
        )

        return aligned_df

    @staticmethod
    def parse_orderbook_side(side_json: str, depth: int = 10) -> List[float]:
        """
        Parses a JSON string representing orderbook bids/asks into a flat list of floats.
        Format: [price1, qty1, price2, qty2, ...]
        """
        try:
            data = json.loads(side_json)
            # Take top 'depth' levels and flatten
            flat = []
            for i in range(min(len(data), depth)):
                flat.extend([float(data[i][0]), float(data[i][1])])
            
            # Padding if less than depth
            while len(flat) < depth * 2:
                flat.extend([0.0, 0.0])
            return flat
        except (json.JSONDecodeError, TypeError):
            return [0.0] * (depth * 2)

    def preprocess_for_rl(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Preprocesses the aligned dataframe for RL observations.
        Vectorizes the JSON parsing of orderbook data.
        """
        # Note: Polars mapping for complex JSON can be slow, 
        # ideally we'd have parsed this during collection or as a pre-processing step.
        # For now, we'll do it here.
        
        print("[*] Parsing orderbook JSON fields (this may take a moment)...")
        
        # We use map_elements for simplicity here, but in production we'd use 
        # a more vectorized approach if possible.
        bids_parsed = df["bids"].map_elements(lambda x: self.parse_orderbook_side(x), return_dtype=pl.List(pl.Float64))
        asks_parsed = df["asks"].map_elements(lambda x: self.parse_orderbook_side(x), return_dtype=pl.List(pl.Float64))

        df = df.with_columns([
            bids_parsed.alias("bids_vec"),
            asks_parsed.alias("asks_vec")
        ])

        return df.drop(["bids", "asks"])

if __name__ == "__main__":
    # Quick test if data exists
    loader = TradingDataLoader()
    try:
        data = loader.load_aligned_data()
        print(f"Loaded {len(data)} aligned records.")
        processed = loader.preprocess_for_rl(data.head(100))
        print("Sample processed data structure:")
        print(processed.head())
    except Exception as e:
        print(f"Data not found or error: {e}")
