import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Setup
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_model_comparison():
    print("--- Generating comparison chart (Classic vs. DL) ---")
    
    results = {
        'Model': ['Random Chance', 'Logistic Regression', 'Random Forest', 'BERT (Deep Learning)'],
        'Accuracy': [50.0, 49.2, 51.5, 57.0], 
        'Type': ['Baseline', 'Classic ML', 'Classic ML', 'Deep Learning']
    }
    
    df = pd.DataFrame(results)
    
    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Balkendiagramm
    colors = ['gray', 'lightblue', 'lightblue', 'green']
    barplot = sns.barplot(x='Model', y='Accuracy', data=df, palette=colors)
    
    plt.axhline(50, color='red', linestyle='--', label='Random Guessing (50%)')
    
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'center', 
                         xytext = (0, 9), 
                         textcoords = 'offset points',
                         fontsize=12, fontweight='bold')

    plt.ylim(45, 60)
    plt.title('Model Comparison: Can News Predict Stocks?', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('', fontsize=12)
    plt.legend()
    
    save_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(save_path)
    print(f"✅ Comparison chart saved to: {save_path}")

if __name__ == "__main__":
    plot_model_comparison()
