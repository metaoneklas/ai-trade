import polars as pl
import json
from pathlib import Path

# Paths to your raw data folders
TRADE_DIR = "data/trades"
OB_DIR = "data/orderbook"
OUTPUT_FILE = "data/processed_rl_data.parquet"

def parse_side(side_json: str, depth: int = 10):
    """Fast JSON string to flat list parser."""
    try:
        data = json.loads(side_json)
        flat = []
        for i in range(min(len(data), depth)):
            flat.extend([float(data[i][0]), float(data[i][1])])
        while len(flat) < depth * 2:
            flat.extend([0.0, 0.0])
        return flat
    except:
        return [0.0] * (depth * 2)

def main():
    print("[*] Loading raw Parquet files...")
    trades_files = list(Path(TRADE_DIR).glob("*.parquet"))
    ob_files = list(Path(OB_DIR).glob("*.parquet"))

    if not trades_files or not ob_files:
        raise FileNotFoundError("Make sure your raw parquet files are in data/trades and data/orderbook")

    trades_df = pl.read_parquet(trades_files)
    ob_df = pl.read_parquet(ob_files)

    print("[*] Parsing ISO Timestamps and Sorting...")
    # FIXED: Changed .%f to %.f to resolve ChronoFormatWarning
    trades_df = trades_df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
    ).drop_nulls("timestamp").sort("timestamp")

    ob_df = ob_df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f", strict=False)
    ).drop_nulls("timestamp").sort("timestamp")

    print("[*] Aligning Streams (join_asof)...")
    df = trades_df.join_asof(ob_df, on="timestamp", strategy="backward").drop_nulls()

    print("[*] Parsing Orderbook JSON to arrays (this may take a few minutes)...")
    df = df.with_columns([
        pl.col("bids").map_elements(lambda x: parse_side(x), return_dtype=pl.List(pl.Float64)).alias("bids_vec"),
        pl.col("asks").map_elements(lambda x: parse_side(x), return_dtype=pl.List(pl.Float64)).alias("asks_vec")
    ])

    print("[*] Engineering RL Features (Returns, Spreads, Imbalances)...")
    df = df.with_columns([
        # Log Returns (Micro-price target)
        (pl.col("price").log().diff()).alias("log_return"),
        
        # Relative Spread
        ((pl.col("asks_vec").list.get(0) - pl.col("bids_vec").list.get(0)) / pl.col("price")).alias("rel_spread"),
        
        # Orderbook Imbalance (Resting Liquidity)
        # FIXED: Added .implode() to resolve DeprecationWarning for list.gather
        (pl.col("bids_vec").list.gather(pl.int_range(1, 20, 2).implode()).list.sum().alias("bid_vol")),
        (pl.col("asks_vec").list.gather(pl.int_range(1, 20, 2).implode()).list.sum().alias("ask_vol")),

        # Trade Flow Imbalance (Aggressive Taker Liquidity)
        pl.when(pl.col("is_buyer_maker") == False).then(pl.col("quantity")).otherwise(0.0).alias("taker_buy_vol"),
        pl.when(pl.col("is_buyer_maker") == True).then(pl.col("quantity")).otherwise(0.0).alias("taker_sell_vol")
    ])

    print("[*] Finalizing Feature Scaling...")
    df = df.with_columns([
        ((pl.col("bid_vol") - pl.col("ask_vol")) / (pl.col("bid_vol") + pl.col("ask_vol") + 1e-8)).alias("ob_imbalance"),
        ((pl.col("taker_buy_vol") - pl.col("taker_sell_vol")) / (pl.col("taker_buy_vol") + pl.col("taker_sell_vol") + 1e-8)).alias("trade_imbalance")
    ])

    # Select only the features we need for the environment
    df = df.select([
        "timestamp", "price", "log_return", "rel_spread", 
        "ob_imbalance", "trade_imbalance"
    ]).drop_nulls()
    
    df.write_parquet(OUTPUT_FILE)
    print(f"[+] Done! Saved {len(df)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()