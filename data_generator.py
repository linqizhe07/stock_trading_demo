"""
模拟数据生成器
生成K线数据（OHLCV）+ 新闻情感分数
"""
import numpy as np
import pandas as pd


def generate_kline_data(
    n_days: int = 1000,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成模拟K线数据（日频OHLCV）

    使用几何布朗运动模拟价格，叠加均值回复和动量效应，
    使其具备一定的可预测结构，方便RL学到有意义的策略。

    Parameters
    ----------
    n_days : int
        生成天数
    initial_price : float
        初始价格
    volatility : float
        日波动率
    trend : float
        日均漂移率
    seed : int
        随机种子
    """
    rng = np.random.default_rng(seed)

    closes = np.zeros(n_days)
    closes[0] = initial_price

    # 生成带有均值回复 + 动量的收盘价
    momentum = 0.0
    for i in range(1, n_days):
        # 均值回复：偏离均线时会有回拉力
        mean_revert = -0.001 * (closes[i - 1] - initial_price) / initial_price
        # 动量：短期趋势延续
        momentum = 0.5 * momentum + rng.normal(0, volatility)
        # 合成收益率
        ret = trend + mean_revert + momentum + rng.normal(0, volatility * 0.5)
        closes[i] = closes[i - 1] * (1 + ret)

    # 根据收盘价生成 OHLV
    highs = closes * (1 + rng.uniform(0, 0.02, n_days))
    lows = closes * (1 - rng.uniform(0, 0.02, n_days))
    opens = lows + rng.uniform(0.3, 0.7, n_days) * (highs - lows)
    volumes = rng.lognormal(mean=15, sigma=0.5, size=n_days).astype(int)

    # 波动放大时成交量也放大
    vol_factor = np.abs(closes / np.roll(closes, 1) - 1)
    vol_factor[0] = 0
    volumes = (volumes * (1 + vol_factor * 20)).astype(int)

    dates = pd.bdate_range(start="2020-01-02", periods=n_days)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": volumes,
        }
    )
    return df


def generate_news_sentiment(
    kline_df: pd.DataFrame,
    noise_level: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成模拟新闻情感分数

    情感分数与未来收益有一定相关性（模拟真实中新闻的预测能力），
    同时加入噪声，避免RL直接靠情感分数"作弊"。

    Parameters
    ----------
    kline_df : pd.DataFrame
        K线数据
    noise_level : float
        噪声水平，越大新闻信号越弱
    seed : int
        随机种子

    Returns
    -------
    pd.DataFrame
        包含 date 和 sentiment 列的DataFrame
        sentiment in [-1, 1]: 负面 -> 正面
    """
    rng = np.random.default_rng(seed)
    closes = kline_df["close"].values
    n = len(closes)

    # 未来收益信号（用未来1-3天的收益作为"领先指标"）
    future_returns = np.zeros(n)
    for i in range(n - 3):
        future_returns[i] = (closes[i + 3] - closes[i]) / closes[i]

    # 归一化到 [-1, 1]
    signal = np.clip(future_returns * 50, -1, 1)

    # 加噪声
    noise = rng.normal(0, noise_level, n)
    sentiment = np.clip(signal + noise, -1, 1)

    # 模拟"并非每天都有新闻"：约30%的天没有新闻，设为0
    no_news_mask = rng.random(n) < 0.3
    sentiment[no_news_mask] = 0.0

    df = pd.DataFrame(
        {
            "date": kline_df["date"],
            "sentiment": np.round(sentiment, 4),
        }
    )
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加常用技术指标

    Parameters
    ----------
    df : pd.DataFrame
        K线数据

    Returns
    -------
    pd.DataFrame
        添加了技术指标的DataFrame
    """
    df = df.copy()
    close = df["close"]

    # 移动平均线
    df["ma5"] = close.rolling(5).mean()
    df["ma20"] = close.rolling(20).mean()

    # RSI (14日)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 波动率 (20日收益标准差)
    df["volatility"] = close.pct_change().rolling(20).std()

    # 成交量变化率
    df["volume_change"] = df["volume"].pct_change()

    return df


if __name__ == "__main__":
    # 快速测试
    kline = generate_kline_data(500)
    kline = add_technical_indicators(kline)
    sentiment = generate_news_sentiment(kline)
    print("K-line shape:", kline.shape)
    print("Sentiment shape:", sentiment.shape)
    print("\nK-line columns:", list(kline.columns))
    print("\nK-line sample:\n", kline.tail(3))
    print("\nSentiment sample:\n", sentiment.tail(3))
