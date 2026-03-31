"""
自定义 Gymnasium 交易环境

State:  K线技术指标 + 新闻情感 + 持仓信息
Action: 连续动作 [-1, 1]，表示目标仓位（-1做空, 0空仓, 1满仓）
Reward: 基于风险调整收益的奖励函数
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_generator import (
    add_technical_indicators,
    generate_kline_data,
    generate_news_sentiment,
)


class StockTradingEnv(gym.Env):
    """
    单股票交易环境，支持做多做空

    Observation Space (维度=12):
        0: close_norm       - 归一化收盘价（相对于初始价格）
        1: ma5_ratio        - MA5 / close - 1
        2: ma20_ratio       - MA20 / close - 1
        3: rsi_norm         - RSI / 100 (归一化到 [0,1])
        4: macd_norm        - 归一化 MACD
        5: macd_signal_norm - 归一化 MACD 信号线
        6: volatility       - 20日波动率
        7: volume_norm      - 归一化成交量变化率
        8: sentiment        - 新闻情感分数 [-1, 1]
        9: position         - 当前仓位 [-1, 1]
        10: unrealized_pnl  - 未实现盈亏比例
        11: holding_days    - 持仓天数 (归一化)

    Action Space:
        连续 [-1, 1]: 目标仓位比例
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        kline_df=None,
        sentiment_df=None,
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,  # 千分之一手续费
        lookback_window: int = 30,
        max_steps: int = None,
    ):
        super().__init__()

        # 生成或使用提供的数据
        if kline_df is None:
            kline_df = generate_kline_data(1000)
        kline_df = add_technical_indicators(kline_df)

        if sentiment_df is None:
            sentiment_df = generate_news_sentiment(kline_df)

        # 合并数据并去除NaN行（技术指标前几行会有NaN）
        self.data = kline_df.merge(sentiment_df, on="date", how="left")
        self.data = self.data.dropna().reset_index(drop=True)

        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.max_steps = max_steps or (len(self.data) - 1)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 交易记录（用于回测可视化）
        self.trade_history = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0  # 当前仓位比例 [-1, 1]
        self.entry_price = 0.0
        self.holding_days = 0
        self.total_reward = 0.0
        self.portfolio_values = [self.initial_balance]
        self.trade_history = []

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        """构建当前观测向量"""
        row = self.data.iloc[self.current_step]
        close = row["close"]
        initial_close = self.data.iloc[self.lookback_window]["close"]

        # 归一化特征
        close_norm = close / initial_close - 1.0
        ma5_ratio = (row["ma5"] / close - 1.0) if close > 0 else 0.0
        ma20_ratio = (row["ma20"] / close - 1.0) if close > 0 else 0.0
        rsi_norm = row["rsi"] / 100.0
        macd_norm = row["macd"] / close if close > 0 else 0.0
        macd_signal_norm = row["macd_signal"] / close if close > 0 else 0.0
        volatility = min(row["volatility"], 0.1) if not np.isnan(row["volatility"]) else 0.0
        volume_norm = np.clip(row["volume_change"], -2, 2) if not np.isnan(row["volume_change"]) else 0.0
        sentiment = row["sentiment"]

        # 持仓信息
        unrealized_pnl = 0.0
        if abs(self.position) > 0.01 and self.entry_price > 0:
            unrealized_pnl = self.position * (close - self.entry_price) / self.entry_price

        holding_norm = min(self.holding_days / 30.0, 1.0)

        obs = np.array(
            [
                close_norm,
                ma5_ratio,
                ma20_ratio,
                rsi_norm,
                macd_norm,
                macd_signal_norm,
                volatility,
                volume_norm,
                sentiment,
                self.position,
                unrealized_pnl,
                holding_norm,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        target_position = float(np.clip(action[0], -1.0, 1.0))
        current_close = self.data.iloc[self.current_step]["close"]

        # ---- 执行交易 ----
        position_change = target_position - self.position
        trade_cost = abs(position_change) * self.transaction_cost * self.balance

        # 更新仓位
        old_position = self.position
        self.position = target_position

        if abs(position_change) > 0.01:
            self.entry_price = current_close
            self.holding_days = 0
            self.trade_history.append(
                {
                    "step": self.current_step,
                    "date": self.data.iloc[self.current_step]["date"],
                    "price": current_close,
                    "old_pos": old_position,
                    "new_pos": target_position,
                    "cost": trade_cost,
                }
            )
        else:
            self.holding_days += 1

        # ---- 计算收益 ----
        self.current_step += 1
        next_close = self.data.iloc[self.current_step]["close"]
        price_return = (next_close - current_close) / current_close

        # 仓位收益 - 交易成本
        portfolio_return = self.position * price_return
        self.balance = self.balance * (1 + portfolio_return) - trade_cost

        self.portfolio_values.append(self.balance)

        # ---- Reward 设计 ----
        # 基础：portfolio return
        reward = portfolio_return * 100  # 放大便于学习

        # 惩罚交易成本（鼓励减少不必要交易）
        reward -= abs(position_change) * 0.1

        # 风险惩罚：如果波动太大，给负奖励
        if len(self.portfolio_values) > 20:
            recent = np.array(self.portfolio_values[-20:])
            recent_returns = np.diff(recent) / recent[:-1]
            if np.std(recent_returns) > 0.03:
                reward -= 0.05

        self.total_reward += reward

        # ---- 终止条件 ----
        terminated = False
        truncated = False

        # 爆仓
        if self.balance < self.initial_balance * 0.5:
            terminated = True
            reward -= 10.0

        # 达到最大步数
        if self.current_step >= len(self.data) - 1:
            truncated = True

        if self.current_step >= self.lookback_window + self.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {
            "balance": self.balance,
            "position": self.position,
            "total_reward": self.total_reward,
            "step": self.current_step,
        }

        return obs, reward, terminated, truncated, info

    def get_portfolio_stats(self) -> dict:
        """计算回测统计指标"""
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(252)

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)

        # 胜率
        winning_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            "total_return": f"{total_return:.2%}",
            "annual_return": f"{annual_return:.2%}",
            "sharpe_ratio": f"{sharpe:.3f}",
            "max_drawdown": f"{max_drawdown:.2%}",
            "win_rate": f"{win_rate:.2%}",
            "total_trades": len(self.trade_history),
            "final_balance": f"{values[-1]:,.2f}",
        }
