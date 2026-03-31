#!/usr/bin/env python3
"""
SAC 强化学习量化交易 Demo

完整流程：生成模拟数据 → 训练SAC → 回测 → 可视化

Usage:
    python main.py                    # 使用默认参数运行
    python main.py --timesteps 100000 # 指定训练步数
    python main.py --seed 123         # 指定随机种子
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="SAC Trading Agent Demo")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="总训练步数 (default: 50000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="学习率 (default: 3e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="批大小 (default: 256)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (default: 42)",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=800,
        help="训练数据天数 (default: 800)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=200,
        help="测试数据天数 (default: 200)",
    )
    args = parser.parse_args()

    # ================================================================
    # Step 1: 生成数据
    # ================================================================
    print("\n" + "=" * 60)
    print("  SAC 强化学习量化交易 Demo")
    print("=" * 60)

    from data_generator import generate_kline_data, generate_news_sentiment, add_technical_indicators

    print("\n[Step 1] 生成模拟数据...")
    total_days = args.train_days + args.test_days
    kline = generate_kline_data(n_days=total_days, seed=args.seed)
    sentiment = generate_news_sentiment(kline, seed=args.seed)
    kline_with_ta = add_technical_indicators(kline)
    print(f"  总数据量: {total_days} 天")
    print(f"  训练集: {args.train_days} 天")
    print(f"  测试集: {args.test_days} 天")
    print(f"  价格范围: {kline['close'].min():.2f} ~ {kline['close'].max():.2f}")
    print(f"  技术指标: MA5, MA20, RSI, MACD, Volatility")
    print(f"  新闻情感: {(sentiment['sentiment'] != 0).sum()} 天有新闻数据")

    # ================================================================
    # Step 2: 训练
    # ================================================================
    from train import train_sac

    print("\n" + "=" * 60)
    start_time = time.time()
    model, train_env, test_env = train_sac(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        save_path="models/sac_trader",
    )
    train_time = time.time() - start_time
    print(f"训练耗时: {train_time:.1f} 秒")

    # ================================================================
    # Step 3: 回测
    # ================================================================
    from backtest import run_backtest, run_baseline_backtest, plot_results
    from train import create_train_test_envs

    print("\n" + "=" * 60)
    print("回测评估")
    print("=" * 60)

    # 在测试集上回测
    _, test_env_bt = create_train_test_envs(
        train_days=args.train_days,
        test_days=args.test_days,
        seed=args.seed,
    )

    # Buy & Hold 基线
    _, test_env_bh = create_train_test_envs(
        train_days=args.train_days,
        test_days=args.test_days,
        seed=args.seed,
    )

    print("\n[Step 3a] SAC 策略回测...")
    result = run_backtest(model, test_env_bt)

    print("\n[Step 3b] Buy & Hold 基线回测...")
    baseline_values = run_baseline_backtest(test_env_bh)

    # 打印统计
    print("\n" + "-" * 40)
    print("  SAC 策略统计")
    print("-" * 40)
    for k, v in result["stats"].items():
        print(f"  {k:>20s}: {v}")

    # Buy & Hold 统计
    bh_return = baseline_values[-1] / baseline_values[0] - 1
    print(f"\n  {'Buy&Hold Return':>20s}: {bh_return:.2%}")

    # ================================================================
    # Step 4: 可视化
    # ================================================================
    print("\n[Step 4] 生成可视化图表...")
    chart_path = plot_results(
        result,
        baseline_values=baseline_values,
        save_path="results/backtest.png",
    )

    # 保存结果到 JSON
    os.makedirs("results", exist_ok=True)
    results_json = {
        "config": {
            "timesteps": args.timesteps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "train_days": args.train_days,
            "test_days": args.test_days,
        },
        "stats": result["stats"],
        "baseline_return": f"{bh_return:.2%}",
        "train_time_seconds": round(train_time, 1),
    }
    with open("results/metrics.json", "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"指标已保存至: results/metrics.json")

    print("\n" + "=" * 60)
    print("  Demo 完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  模型:   models/sac_trader.zip")
    print(f"  图表:   results/backtest.png")
    print(f"  指标:   results/metrics.json")


if __name__ == "__main__":
    main()
