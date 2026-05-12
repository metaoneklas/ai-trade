# 🤖 AI-RL Trading Agent

Automated cryptocurrency trading using Reinforcement Learning (RL) and High-Frequency data from Binance.

## 🚀 Project Goal
Build, optimize, and deploy an RL agent capable of profitable, risk-adjusted autonomous trading, fully accounting for exchange fees and slippage.

## 📂 Project Structure
- `data/`: Parquet files storage.
  - `raw/`: Untouched data from `gather.py`.
  - `processed/`: Cleaned and normalized data.
  - `features/`: Engineered features for the RL agent.
- `src/`: Source code.
  - `data/`: Data loading and preprocessing logic (using `polars`).
  - `environment/`: Custom Gymnasium trading environment.
  - `models/`: RL model definitions and training logic (Stable-Baselines3).
  - `utils/`: Common utilities and helpers.
- `tests/`: Unit and integration tests.
- `notebooks/`: Exploratory Data Analysis (EDA) and research.
- `learn.py`: Script to build and train the RL environment.
- `start.py`: Entry point for testing and executing the bot.
- `gather.py`: Binance websocket data collection script.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Data Handling:** `polars`, `numpy`
- **RL Framework:** `gymnasium`, `stable-baselines3`
- **Data Source:** Binance Websockets (AggTrades & Depth10)

## 📋 Quantitative Standards
Refer to `GEMINI.md` for architectural mandates, including memory efficiency, vectorization, and financial reality constraints.
