import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setup Pfade
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_djia_data, load_btc_data
from src.model import BERTSentimentClassifier 

# --- KONFIGURATION ---
MODEL_NAME = 'ProsusAI/finbert'  
MAX_LEN = 160
BATCH_SIZE = 4
EPOCHS = 4
LEARNING_RATE = 2e-5

#  ML TRICK: DATA AUGMENTATION 
def augment_data_with_analyst_ratings(original_df):
    """
    Erzeugt synthetische Trainingsdaten für Analysten-Sprache.
    Das bringt dem Modell bei, dass Wörter wie 'Downgrade' oder 'Sell' wichtiger sind
    als Banknamen wie 'Goldman Sachs'.
    """
    print("💉 Injiziere synthetische Analysten-Daten (Data Augmentation)...")
    
    # Muster für klare Signale
    data = []
    
    # 1. Bearish Examples (Muss als DOWN/0 gelernt werden)
    bearish_templates = [
        "{bank} downgrades {stock} to Sell",
        "{bank} cuts target price for {stock}",
        "{stock} receives a Sell rating from {bank}",
        "Investors should sell {stock} immediately",
        "{bank} starts coverage on {stock} with Underperform rating",
        "Negative outlook for {stock} due to headwinds",
        "{stock} shares tumble after downgrade by {bank}",
        "Sell {stock} before it crashes further",
        "{bank} issues a warning on {stock} earnings",
        "Analyst rating for {stock} lowered to underweight"
    ]
    
    # 2. Bullish Examples (Muss als UP/1 gelernt werden)
    bullish_templates = [
        "{bank} upgrades {stock} to Buy",
        "{bank} raises target price for {stock}",
        "{stock} is a top pick for {bank}",
        "Strong buy signal for {stock}",
        "{bank} starts coverage on {stock} with Outperform rating",
        "Positive outlook for {stock} as demand grows",
        "{stock} shares rally after upgrade",
        "Buy the dip on {stock}",
        "Record earnings expected for {stock}",
        "Analyst rating for {stock} raised to overweight"
    ]
    
    banks = ["Goldman Sachs", "JPMorgan", "Morgan Stanley", "Bank of America", "Citi", "Deutsche Bank"]
    stocks = ["Apple", "Tesla", "Nvidia", "AMD", "Microsoft", "Amazon", "Intel", "Super Micro"]
    
    # Wir generieren zufällige Kombinationen
    import random
    
    # Wir erzeugen 200 zusätzliche Trainingsbeispiele (Genug, damit BERT es lernt)
    for _ in range(100):
        # Bearish (Label 0)
        tmpl = random.choice(bearish_templates)
        text = tmpl.format(bank=random.choice(banks), stock=random.choice(stocks))
        data.append({'text': text, 'label': 0})
        
        # Bullish (Label 1)
        tmpl = random.choice(bullish_templates)
        text = tmpl.format(bank=random.choice(banks), stock=random.choice(stocks))
        data.append({'text': text, 'label': 1})
        
    aug_df = pd.DataFrame(data)
    
    # Zusammenfügen
    combined_df = pd.concat([original_df, aug_df], ignore_index=True)
    # Mischen
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Datensatz erweitert: {len(original_df)} -> {len(combined_df)} Zeilen.")
    return combined_df

class CryptoStockDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(data_type='stock'):
    data_path = BASE_DIR / "data"
    
    # Daten laden
    if data_type == 'mixed':
        df_stock = load_djia_data(data_path / "Combined_News_DJIA.csv")
        df_crypto = load_btc_data(data_path / "BTC.csv")
        df = pd.concat([df_stock, df_crypto], ignore_index=True)
    elif data_type == 'stock':
        df = load_djia_data(data_path / "Combined_News_DJIA.csv")
    else:
        df = load_btc_data(data_path / "BTC.csv")

    # Wir erweitern die Daten VOR dem Split
    df = augment_data_with_analyst_ratings(df)

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = CryptoStockDataset(df_train.text.to_numpy(), df_train.label.to_numpy(), tokenizer, MAX_LEN)
    test_dataset = CryptoStockDataset(df_test.text.to_numpy(), df_test.label.to_numpy(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Nutze Apple Silicon GPU (MPS)!")
    else:
        print(f"Nutze Gerät: {device}")
    
    model = BERTSentimentClassifier(model_name=MODEL_NAME, n_classes=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(device)

    print("\nStarte Training mit Data Augmentation...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    print("\nStarte Evaluation...")
    model.eval()
    predictions, real_values = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            real_values.extend(labels.cpu().tolist())

    acc = accuracy_score(real_values, predictions)
    print(f"\n🏆 FINALER SCORE ({data_type.upper()}): Accuracy = {acc:.2f}")
    
    model_save_path = BASE_DIR / "models"
    model_save_path.mkdir(exist_ok=True)
    save_file = model_save_path / "finbert_trained.pth"
    torch.save(model.state_dict(), save_file)
    print(f"💾 Modell erfolgreich gespeichert unter: {save_file}")

if __name__ == "__main__":
    train_model(data_type='stock')