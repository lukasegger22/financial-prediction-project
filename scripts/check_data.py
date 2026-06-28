import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# Define data paths
DATA_DIR = BASE_DIR / "data"
BTC_FILE = DATA_DIR / "BTC.csv"
DJIA_FILE = DATA_DIR / "Combined_News_DJIA.csv"

def check_btc():
    print(f"\n--- Checking BTC file: {BTC_FILE.name} ---")
    if not BTC_FILE.exists():
        print(f"❌ ERROR: File not found at {BTC_FILE}")
        return
    
    try:
        df = pd.read_csv(BTC_FILE)
        print("✅ File loaded.")
        print(f"   Number of rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print("\n   Example row (first entry):")
        print(df.iloc[0])
        
        text_cols = [col for col in df.columns if 'news' in col.lower() or 'text' in col.lower() or 'title' in col.lower()]
        if text_cols:
            print(f"   ℹ️  Possible text columns found: {text_cols}")
        else:
            print("   ⚠️  WARNING: No obvious column with 'news', 'text', or 'title' in its name was found.")
            
    except Exception as e:
        print(f"❌ Critical read error: {e}")

def check_djia():
    print(f"\n--- Checking DJIA file: {DJIA_FILE.name} ---")
    if not DJIA_FILE.exists():
        print(f"❌ ERROR: File not found at {DJIA_FILE}")
        return

    try:
        df = pd.read_csv(DJIA_FILE)
        print("✅ File loaded.")
        print(f"   Number of rows: {len(df)}")
        print(f"   Columns (excerpt): {list(df.columns)[:5]} ...")
    except Exception as e:
        print(f"❌ Critical read error: {e}")

if __name__ == "__main__":
    print("Starting data check...")
    check_btc()
    check_djia()
    print("\nCheck complete.")
