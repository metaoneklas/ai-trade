---
id: TASK-2
title: Implement learn.py for RL Environment Construction
status: Done
assignee: []
created_date: '2026-05-12 05:41'
updated_date: '2026-05-12 05:43'
labels: []
dependencies: []
modified_files:
  - src/data/loader.py
  - src/environment/trading_env.py
  - learn.py
priority: high
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create `learn.py` to build and configure the Reinforcement Learning environment. This script will be responsible for defining the trading environment, loading historical data using `polars`, and preparing the state representation (observations) for the RL agent. It must strictly adhere to the Farama Foundation `gymnasium` API.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Define a `gymnasium.Env` subclass for cryptocurrency trading.
- [x] #2 Implement `reset()` and `step()` methods returning standard Gymnasium outputs.
- [x] #3 Integrate `polars` to load and process Parquet data from `data/` for observations.
- [x] #4 Ensure observation space includes price data (AggTrades) and orderbook depth (Depth10).
- [x] #5 Incorporate exchange fees and slippage into the reward calculation.
- [x] #6 Ensure the script is type-hinted and follows `GEMINI.md` standards.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented `learn.py` and supporting modular components. 
1. `src/data/loader.py`: Uses Polars to load and align AggTrades and Depth10 orderbook Parquet files.
2. `src/environment/trading_env.py`: A custom `gymnasium.Env` that manages a USDT/BTC portfolio, calculates rewards based on net worth changes, and accounts for 0.1% fees and slippage. Observation space (45 features) includes price, quantity, and top 10 orderbook levels.
3. `learn.py`: The training entry point that initializes the environment and sets up a PPO model template using Stable-Baselines3, adhering to quantitative standards.
<!-- SECTION:FINAL_SUMMARY:END -->
