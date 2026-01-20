import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_djia_data(filepath):
    """
    Lädt und bereinigt den Aktien-Datensatz (Kaggle 1).
    """
    print(f"Lade DJIA Daten von {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Datum formatieren
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Text kombinieren (Top1 bis Top25)
    df.fillna('', inplace=True)
    
    df['text'] = df.iloc[:, 2:27].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    # 3. Label ist schon da ('Label' 0 oder 1)
    df.rename(columns={'Label': 'label', 'Date': 'date'}, inplace=True)
    
    return df[['date', 'text', 'label']]

def load_btc_data(filepath):
    """
    Lädt und bereinigt den Bitcoin-Datensatz (Kaggle 2).
    WICHTIG: Erstellt das Label basierend auf dem Preis von MORGEN.
    """
    print(f"Lade BTC Daten von {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Datum formatieren (Spalte heißt 'begins_at')
    df.rename(columns={'begins_at': 'date', 'articles': 'raw_text'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # 2. Label erstellen (Shift!)
    df['next_day_return'] = df['close_price'].pct_change().shift(-1)
    
    # Label: 1 wenn Return positiv, sonst 0
    df['label'] = (df['next_day_return'] > 0).astype(int)
    
    df.dropna(subset=['next_day_return'], inplace=True)
    
    # 3. Text bereinigen
    def clean_article_list(text_list_str):
        if not isinstance(text_list_str, str):
            return ""

        cleaned = text_list_str.replace("['", "").replace("']", "").replace("', '", " ").replace('", "', " ")
        return cleaned

    df['text'] = df['raw_text'].apply(clean_article_list)
    
    return df[['date', 'text', 'label']]

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent / "data"
    
    # Test DJIA
    try:
        df_djia = load_djia_data(base_path / "Combined_News_DJIA.csv")
        print(f"DJIA Success: {len(df_djia)} Zeilen. Beispiel Text: {df_djia.iloc[0]['text'][:50]}...")
    except Exception as e:
        print(f"DJIA Error: {e}")

    # Test BTC
    try:
        df_btc = load_btc_data(base_path / "BTC.csv")
        print(f"BTC Success: {len(df_btc)} Zeilen. Beispiel Text: {df_btc.iloc[0]['text'][:50]}...")
    except Exception as e:
        print(f"BTC Error: {e}")
