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
# NEU: Wir importieren das Modell, statt es hier zu definieren!
from src.model import BERTSentimentClassifier 

# --- KONFIGURATION ---
MODEL_NAME = 'bert-base-uncased' 
MAX_LEN = 160       
BATCH_SIZE = 4      
EPOCHS = 4          
LEARNING_RATE = 2e-5

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

# HIER WURDE DIE KLASSE GELÖSCHT (Refactoring)

def train_model(data_type='mixed'):
    data_path = BASE_DIR / "data"
    
    if data_type == 'mixed':
        print("🌀 Lade ALLES (Aktien + Krypto) für das 'Universal Model'...")
        df_stock = load_djia_data(data_path / "Combined_News_DJIA.csv")
        df_crypto = load_btc_data(data_path / "BTC.csv")
        df = pd.concat([df_stock, df_crypto], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"✅ Kombinierter Datensatz: {len(df)} Zeilen.")
        
    elif data_type == 'stock':
        df = load_djia_data(data_path / "Combined_News_DJIA.csv")
        print("Trainiere nur auf AKTIEN Daten")
    else:
        df = load_btc_data(data_path / "BTC.csv")
        print("Trainiere nur auf BITCOIN Daten")

    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = CryptoStockDataset(df_train.text.to_numpy(), df_train.label.to_numpy(), tokenizer, MAX_LEN)
    test_dataset = CryptoStockDataset(df_test.text.to_numpy(), df_test.label.to_numpy(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Nutze Apple Silicon GPU (MPS)!")
    else:
        print(f"Nutze Gerät: {device}")
    
    # NEU: Wir übergeben den MODEL_NAME an die importierte Klasse
    model = BERTSentimentClassifier(model_name=MODEL_NAME, n_classes=2).to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss().to(device)

    print("\nStarte Training...")
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

if __name__ == "__main__":
    train_model(data_type='mixed')