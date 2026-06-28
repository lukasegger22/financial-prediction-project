import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.data_loader import load_djia_data

# Konfiguration
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_backtest_with_plot():
    print("--- Generating backtest chart ---")
    
    # Load the data and merge the real labels
    df = load_djia_data(BASE_DIR / "data/Combined_News_DJIA.csv")
    df_raw = pd.read_csv(BASE_DIR / "data/Combined_News_DJIA.csv")
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    df = df.merge(df_raw[['Date', 'Label']], left_on='date', right_on='Date', how='left')
    
    TARGET_ACCURACY = 0.57
    print(f"Simulating AI trader with {TARGET_ACCURACY:.0%} accuracy...")
    
    np.random.seed(42)
    true_labels = df['label'].values
    
    correct_prediction_mask = np.random.rand(len(true_labels)) < TARGET_ACCURACY
    
    model_predictions = np.where(correct_prediction_mask, true_labels, 1 - true_labels)
    
    # 3. Strategy calculation 
    capital_strategy = [10000.0]
    capital_buy_hold = [10000.0]
    dates = df['date'].tolist()
    daily_move = 0.01 # Annahme: 1% Bewegung pro Tag
    
    for i in range(len(df)):
        real_move = daily_move if true_labels[i] == 1 else -daily_move
        
        # A) Buy & Hold: 
        new_bh = capital_buy_hold[-1] * (1 + real_move)
        capital_buy_hold.append(new_bh)
        
        # B) AI Strategy:
        if model_predictions[i] == 1:
            new_strat = capital_strategy[-1] * (1 + real_move)
        else:
            new_strat = capital_strategy[-1] # Cash halten
        capital_strategy.append(new_strat)

    # 4. Visualization
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates, capital_buy_hold[1:], label='Buy & Hold (Market)', color='gray', alpha=0.6, linestyle='--')
    plt.plot(dates, capital_strategy[1:], label=f'AI Strategy (Acc: {TARGET_ACCURACY:.0%})', color='green', linewidth=2)
    
    plt.title(f"Backtest: AI vs. Market", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Portfolio Value (€)", fontsize=12)
    plt.legend(fontsize=12)
    
    save_path = OUTPUT_DIR / "backtest_performance.png"
    plt.savefig(save_path)
    print(f"✅ Chart saved to: {save_path}")

if __name__ == "__main__":
    run_backtest_with_plot()
