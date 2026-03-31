"""
回测与可视化模块

对训练好的SAC模型在测试集上进行回测，并生成可视化图表
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import SAC

from data_generator import generate_kline_data, generate_news_sentiment
from trading_env import StockTradingEnv


def run_backtest(model, env) -> dict:
    """
    在环境上运行回测

    Returns
    -------
    dict
        包含交易历史、资产曲线、统计指标的字典
    """
    obs, _ = env.reset()
    done = False
    positions = []
    dates = []
    prices = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 获取底层环境（跳过 Monitor wrapper）
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        step = base_env.current_step
        row = base_env.data.iloc[step]
        positions.append(base_env.position)
        dates.append(row["date"])
        prices.append(row["close"])

    # 获取统计数据
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    stats = base_env.get_portfolio_stats()
    portfolio_values = base_env.portfolio_values
    trade_history = base_env.trade_history

    return {
        "stats": stats,
        "portfolio_values": portfolio_values,
        "positions": positions,
        "dates": dates,
        "prices": prices,
        "trade_history": trade_history,
    }


def run_baseline_backtest(env) -> list:
    """
    Buy-and-hold 基线策略

    Returns
    -------
    list
        资产曲线
    """
    obs, _ = env.reset()

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    initial_price = base_env.data.iloc[base_env.current_step]["close"]
    initial_balance = base_env.initial_balance

    values = [initial_balance]
    done = False

    while not done:
        # 始终满仓做多
        action = np.array([1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step = base_env.current_step
        current_price = base_env.data.iloc[step]["close"]
        values.append(initial_balance * current_price / initial_price)

    return values


def plot_results(
    backtest_result: dict,
    baseline_values: list = None,
    save_path: str = "results/backtest.png",
):
    """
    生成回测可视化图表

    包含4个子图:
    1. 股价走势 + 交易信号
    2. 资产曲线 (SAC vs Buy&Hold)
    3. 仓位变化
    4. 回撤曲线
    """
    dates = backtest_result["dates"]
    prices = backtest_result["prices"]
    positions = backtest_result["positions"]
    portfolio = backtest_result["portfolio_values"]
    trades = backtest_result["trade_history"]
    stats = backtest_result["stats"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(
        "SAC Trading Agent - Backtest Results",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # ---- 1. 股价 + 交易点 ----
    ax1 = axes[0]
    ax1.plot(dates, prices, "k-", linewidth=1, alpha=0.8, label="Close Price")

    # 标记买卖点
    for trade in trades:
        if trade["date"] in dates:
            color = "green" if trade["new_pos"] > trade["old_pos"] else "red"
            marker = "^" if trade["new_pos"] > trade["old_pos"] else "v"
            ax1.scatter(
                trade["date"],
                trade["price"],
                c=color,
                marker=marker,
                s=50,
                zorder=5,
                alpha=0.7,
            )

    ax1.set_ylabel("Price")
    ax1.set_title("Stock Price & Trade Signals (▲ Buy  ▼ Sell)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ---- 2. 资产曲线 ----
    ax2 = axes[1]
    # SAC资产曲线对齐日期
    sac_len = min(len(portfolio), len(dates) + 1)
    sac_dates = [dates[0]] + list(dates[: sac_len - 1]) if sac_len > 1 else dates[:sac_len]

    ax2.plot(
        sac_dates[:sac_len],
        portfolio[:sac_len],
        "b-",
        linewidth=1.5,
        label=f"SAC Agent ({stats['total_return']})",
    )

    if baseline_values is not None:
        bl_len = min(len(baseline_values), len(sac_dates))
        bh_return = (baseline_values[bl_len - 1] / baseline_values[0] - 1)
        ax2.plot(
            sac_dates[:bl_len],
            baseline_values[:bl_len],
            "r--",
            linewidth=1,
            alpha=0.7,
            label=f"Buy & Hold ({bh_return:.2%})",
        )

    ax2.set_ylabel("Portfolio Value")
    ax2.set_title("Portfolio Value Comparison")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=portfolio[0], color="gray", linestyle=":", alpha=0.5)

    # ---- 3. 仓位变化 ----
    ax3 = axes[2]
    ax3.fill_between(
        dates,
        positions,
        0,
        where=np.array(positions) >= 0,
        alpha=0.4,
        color="green",
        label="Long",
    )
    ax3.fill_between(
        dates,
        positions,
        0,
        where=np.array(positions) < 0,
        alpha=0.4,
        color="red",
        label="Short",
    )
    ax3.plot(dates, positions, "k-", linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("Position")
    ax3.set_title("Position Over Time")
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    # ---- 4. 回撤曲线 ----
    ax4 = axes[3]
    pv = np.array(portfolio[:sac_len])
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / peak * 100

    ax4.fill_between(
        sac_dates[:sac_len],
        drawdown,
        0,
        alpha=0.4,
        color="red",
    )
    ax4.plot(sac_dates[:sac_len], drawdown, "r-", linewidth=0.5)
    ax4.set_ylabel("Drawdown (%)")
    ax4.set_title(f"Drawdown (Max: {stats['max_drawdown']})")
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()

    # x轴日期格式
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 添加统计信息文字
    stats_text = (
        f"Sharpe: {stats['sharpe_ratio']}  |  "
        f"Win Rate: {stats['win_rate']}  |  "
        f"Trades: {stats['total_trades']}  |  "
        f"Final: ${stats['final_balance']}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=11, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"回测图表已保存至: {save_path}")

    return save_path
