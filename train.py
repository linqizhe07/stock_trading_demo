"""
SAC 训练脚本

使用 Stable-Baselines3 的 SAC 算法训练交易智能体
"""
import os
import sys
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from data_generator import generate_kline_data, generate_news_sentiment
from trading_env import StockTradingEnv


class TradingMetricsCallback(BaseCallback):
    """训练过程中记录交易指标的回调"""

    def __init__(self, eval_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # 获取最近的 episode 信息
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean(
                    [ep["r"] for ep in self.model.ep_info_buffer]
                )
                mean_len = np.mean(
                    [ep["l"] for ep in self.model.ep_info_buffer]
                )
                print(
                    f"  Step {self.n_calls:>7d} | "
                    f"Mean Reward: {mean_reward:>8.2f} | "
                    f"Mean Episode Len: {mean_len:>6.0f}"
                )
        return True


def create_train_test_envs(
    train_days: int = 800,
    test_days: int = 200,
    seed: int = 42,
):
    """创建训练和测试环境（不同的数据段）"""
    # 生成完整数据集
    kline = generate_kline_data(n_days=train_days + test_days, seed=seed)
    sentiment = generate_news_sentiment(kline, seed=seed)

    # 分割
    train_kline = kline.iloc[:train_days].reset_index(drop=True)
    train_sentiment = sentiment.iloc[:train_days].reset_index(drop=True)

    test_kline = kline.iloc[train_days:].reset_index(drop=True)
    test_sentiment = sentiment.iloc[train_days:].reset_index(drop=True)

    train_env = Monitor(
        StockTradingEnv(
            kline_df=train_kline,
            sentiment_df=train_sentiment,
            transaction_cost=0.001,
        )
    )

    test_env = Monitor(
        StockTradingEnv(
            kline_df=test_kline,
            sentiment_df=test_sentiment,
            transaction_cost=0.001,
        )
    )

    return train_env, test_env


def train_sac(
    total_timesteps: int = 50_000,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    seed: int = 42,
    save_path: str = "models/sac_trader",
):
    """
    训练 SAC 交易智能体

    Parameters
    ----------
    total_timesteps : int
        总训练步数
    learning_rate : float
        学习率
    batch_size : int
        批大小
    seed : int
        随机种子
    save_path : str
        模型保存路径
    """
    print("=" * 60)
    print("SAC 交易智能体训练")
    print("=" * 60)

    # 创建环境
    print("\n[1/3] 创建训练/测试环境...")
    train_env, test_env = create_train_test_envs(seed=seed)
    print(f"  训练环境观测空间: {train_env.observation_space.shape}")
    print(f"  训练环境动作空间: {train_env.action_space.shape}")

    # 创建 SAC 模型
    print("\n[2/3] 初始化 SAC 模型...")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        buffer_size=100_000,
        learning_starts=1000,
        tau=0.005,          # 软更新系数
        gamma=0.99,         # 折扣因子
        ent_coef="auto",    # 自动调节熵系数（SAC的核心）
        target_entropy="auto",
        policy_kwargs=dict(
            net_arch=[256, 256],  # 两层256的隐藏层
        ),
        verbose=0,
        seed=seed,
    )

    print(f"  网络结构: MLP [256, 256]")
    print(f"  学习率: {learning_rate}")
    print(f"  批大小: {batch_size}")
    print(f"  总训练步数: {total_timesteps:,}")

    # 训练
    print(f"\n[3/3] 开始训练 ({total_timesteps:,} steps)...")
    print("-" * 60)

    callback = TradingMetricsCallback(eval_freq=5000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False,
    )

    print("-" * 60)
    print("训练完成!")

    # 保存模型
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"模型已保存至: {save_path}")

    return model, train_env, test_env


if __name__ == "__main__":
    model, train_env, test_env = train_sac(
        total_timesteps=50_000,
        save_path="models/sac_trader",
    )
