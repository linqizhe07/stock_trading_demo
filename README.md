# SAC Reinforcement Learning for Quantitative Trading

A prototype quantitative trading system powered by Soft Actor-Critic (SAC). It trains a trading agent on simulated candlestick (K-line) data combined with news sentiment signals, supports both long and short positions, and includes a full backtesting and visualization pipeline.

## Project Structure

```
rl-trading-demo/
├── main.py              # Entry point — runs the full pipeline
├── data_generator.py    # Simulated data (K-line OHLCV + news sentiment)
├── trading_env.py       # Custom Gymnasium trading environment
├── train.py             # SAC training script (Stable-Baselines3)
├── backtest.py          # Backtesting engine + visualization
├── requirements.txt     # Dependencies
├── models/              # Saved model checkpoints (auto-generated)
└── results/             # Backtest charts and metrics (auto-generated)
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

After training completes, you'll find the backtest chart at `results/backtest.png`, performance metrics at `results/metrics.json`, and the trained model at `models/sac_trader.zip`.

### CLI Options

```bash
python main.py --timesteps 100000              # more training steps (default: 50000)
python main.py --lr 1e-4                       # adjust learning rate (default: 3e-4)
python main.py --seed 123                      # change random seed
python main.py --train-days 1200 --test-days 300  # adjust dataset size
```

## System Design

### Data Layer (data_generator.py)

K-line prices are generated via Geometric Brownian Motion with mean-reversion and momentum overlays, giving the price series a degree of predictable structure that the RL agent can learn to exploit. News sentiment scores are weakly correlated with 1-3 day forward returns (mimicking the limited predictive power of real-world news), and roughly 30% of trading days carry no news signal.

### Environment Layer (trading_env.py)

A Gymnasium-compatible environment with a 12-dimensional observation space:

- Technical indicators: normalized close price, MA5/MA20 ratios, RSI, MACD, volatility, volume change rate
- External signal: news sentiment score in [-1, 1]
- Portfolio state: current position, unrealized PnL, holding duration

The action space is a continuous scalar in [-1, 1], representing the target position ratio (negative for short, positive for long).

Reward shaping balances three factors: position return (primary signal), transaction cost penalty (discourages overtrading), and volatility penalty (encourages risk control).

### Algorithm Layer (train.py)

Built on Stable-Baselines3's SAC implementation with the following configuration:

- Policy network: MLP [256, 256]
- Automatic entropy coefficient tuning (core advantage of SAC)
- Replay buffer: 100K transitions, soft update tau = 0.005

### Backtesting Layer (backtest.py)

Generates a four-panel chart: price action with trade signals, portfolio value comparison (SAC vs. Buy & Hold), position changes over time, and drawdown curve. Reported metrics include annualized return, Sharpe ratio, max drawdown, and win rate.

## Future Extensions

1. **Real market data** — swap `data_generator.py` for a tushare / akshare / yfinance data source
2. **Raw text encoding** — replace the sentiment score with FinBERT embeddings of raw news articles
3. **Multi-asset environment** — extend to portfolio-level allocation across multiple stocks
4. **Advanced reward shaping** — incorporate Sortino ratio, Calmar ratio, etc.
5. **Higher frequency** — move from daily to minute-level bars with order book features

## Requirements

- Python >= 3.9
- PyTorch (installed automatically by stable-baselines3)
- stable-baselines3 >= 2.3
- gymnasium >= 0.29
- numpy, pandas, matplotlib
