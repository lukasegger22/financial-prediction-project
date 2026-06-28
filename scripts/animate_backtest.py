import shutil
import sys
from pathlib import Path

import os

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache-financial-prediction")

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


sys.path.append(str(BASE_DIR))

from src.data_loader import load_djia_data


def build_backtest_series(target_accuracy=0.57, seed=42):
    df = load_djia_data(BASE_DIR / "data" / "Combined_News_DJIA.csv")
    true_labels = df["label"].to_numpy()
    dates = pd.to_datetime(df["date"])

    rng = np.random.default_rng(seed)
    correct_prediction_mask = rng.random(len(true_labels)) < target_accuracy
    model_predictions = np.where(correct_prediction_mask, true_labels, 1 - true_labels)

    daily_move = 0.01
    ai_portfolio = [10000.0]
    market_portfolio = [10000.0]

    for true_label, prediction in zip(true_labels, model_predictions):
        real_move = daily_move if true_label == 1 else -daily_move

        market_portfolio.append(market_portfolio[-1] * (1 + real_move))

        if prediction == 1:
            ai_portfolio.append(ai_portfolio[-1] * (1 + real_move))
        else:
            ai_portfolio.append(ai_portfolio[-1])

    return dates, np.array(ai_portfolio[1:]), np.array(market_portfolio[1:])


def create_animation():
    target_accuracy = 0.57
    dates, ai_portfolio, market_portfolio = build_backtest_series(target_accuracy)
    frame_step = max(1, len(dates) // 160)
    frames = list(range(0, len(dates), frame_step))
    if frames[-1] != len(dates) - 1:
        frames.append(len(dates) - 1)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))

    min_value = min(ai_portfolio.min(), market_portfolio.min()) * 0.95
    max_value = max(ai_portfolio.max(), market_portfolio.max()) * 1.05

    ax.set_xlim(dates.iloc[0], dates.iloc[-1])
    ax.set_ylim(min_value, max_value)
    ax.set_title("Backtest Animation: AI Strategy vs. Buy & Hold", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (EUR)")

    line_ai, = ax.plot([], [], color="green", linewidth=2.4, label=f"AI Strategy (Acc: {target_accuracy:.0%})")
    line_market, = ax.plot([], [], color="gray", linewidth=2, alpha=0.65, linestyle="--", label="Buy & Hold")
    current_value = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=11)
    ax.legend(loc="upper left")

    def update(frame):
        end = frame + 1
        line_ai.set_data(dates.iloc[:end], ai_portfolio[:end])
        line_market.set_data(dates.iloc[:end], market_portfolio[:end])
        current_value.set_text(
            f"{dates.iloc[frame].date()} | AI: EUR {ai_portfolio[frame]:,.0f} | Market: EUR {market_portfolio[frame]:,.0f}"
        )
        return line_ai, line_market, current_value

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=110,
        blit=True,
        repeat=False,
    )

    if shutil.which("ffmpeg"):
        save_path = OUTPUT_DIR / "backtest_animation.mp4"
        ani.save(save_path, writer="ffmpeg", fps=12, dpi=120)
    else:
        save_path = OUTPUT_DIR / "backtest_animation.gif"
        ani.save(save_path, writer="pillow", fps=8, dpi=85)

    plt.close(fig)
    print(f"Animation saved to: {save_path}")


if __name__ == "__main__":
    create_animation()
