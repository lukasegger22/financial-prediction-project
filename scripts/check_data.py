import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

# Pfade zu den Daten definieren
DATA_DIR = BASE_DIR / "data"
BTC_FILE = DATA_DIR / "BTC.csv"
DJIA_FILE = DATA_DIR / "Combined_News_DJIA.csv"

def check_btc():
    print(f"\n--- Prüfe BTC Datei: {BTC_FILE.name} ---")
    if not BTC_FILE.exists():
        print(f"❌ FEHLER: Datei nicht gefunden unter {BTC_FILE}")
        return
    
    try:
        df = pd.read_csv(BTC_FILE)
        print("✅ Datei geladen.")
        print(f"   Anzahl Zeilen: {len(df)}")
        print(f"   Spalten: {list(df.columns)}")
        print("\n   Beispiel-Zeile (Erster Eintrag):")
        print(df.iloc[0])
        
        text_cols = [col for col in df.columns if 'news' in col.lower() or 'text' in col.lower() or 'title' in col.lower()]
        if text_cols:
            print(f"   ℹ️  Mögliche Text-Spalten gefunden: {text_cols}")
        else:
            print("   ⚠️  WARNUNG: Keine offensichtliche Spalte mit 'news', 'text' oder 'title' im Namen gefunden.")
            
    except Exception as e:
        print(f"❌ Kritischer Fehler beim Lesen: {e}")

def check_djia():
    print(f"\n--- Prüfe DJIA Datei: {DJIA_FILE.name} ---")
    if not DJIA_FILE.exists():
        print(f"❌ FEHLER: Datei nicht gefunden unter {DJIA_FILE}")
        return

    try:
        df = pd.read_csv(DJIA_FILE)
        print("✅ Datei geladen.")
        print(f"   Anzahl Zeilen: {len(df)}")
        print(f"   Spalten (Auszug): {list(df.columns)[:5]} ...")
    except Exception as e:
        print(f"❌ Kritischer Fehler beim Lesen: {e}")

if __name__ == "__main__":
    print("Starte Data-Check...")
    check_btc()
    check_djia()
    print("\nCheck abgeschlossen.")
