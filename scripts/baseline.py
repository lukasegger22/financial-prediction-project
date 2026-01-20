import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Setup Pfade
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.data_loader import load_djia_data, load_btc_data

def run_baseline(data_type='stock'):
    print(f"\n--- Starte CLASSIC ML Baseline für {data_type.upper()} ---")
    data_path = BASE_DIR / "data"

    # 1. Daten laden
    if data_type == 'stock':
        df = load_djia_data(data_path / "Combined_News_DJIA.csv")
    elif data_type == 'crypto':
        df = load_btc_data(data_path / "BTC.csv")
    elif data_type == 'mixed':
        df_stock = load_djia_data(data_path / "Combined_News_DJIA.csv")
        df_crypto = load_btc_data(data_path / "BTC.csv")
        df = pd.concat([df_stock, df_crypto], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    # 3. Vektorisierung (TF-IDF statt BERT Embeddings)
    print("Vektoriere Text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Modell A: Logistische Regression (Der Klassiker)
    print("Trainiere Logistic Regression...")
    clf_log = LogisticRegression(max_iter=1000)
    clf_log.fit(X_train_vec, y_train)
    preds_log = clf_log.predict(X_test_vec)
    acc_log = accuracy_score(y_test, preds_log)
    print(f"👉 Logistic Regression Accuracy: {acc_log:.4f}")

    # 5. Modell B: Random Forest (Entscheidungsbäume)
    print("Trainiere Random Forest...")
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train_vec, y_train)
    preds_rf = clf_rf.predict(X_test_vec)
    acc_rf = accuracy_score(y_test, preds_rf)
    print(f"👉 Random Forest Accuracy:      {acc_rf:.4f}")

    return acc_log, acc_rf

if __name__ == "__main__":
    run_baseline('stock')
    run_baseline('crypto')
    # run_baseline('mixed') # Optional
