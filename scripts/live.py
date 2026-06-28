import sys
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.model import BERTSentimentClassifier

# --- CONFIGURATION ---
MODEL_NAME = 'ProsusAI/finbert' 
MODEL_PATH = BASE_DIR / "models" / "finbert_trained.pth"
MAX_LEN = 160

def run_live_prediction():
    print("\n--- 🔴 LIVE AI TRADING SYSTEM ---")
    print("Loading trained model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")

    # 1. Build architecture
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BERTSentimentClassifier(model_name=MODEL_NAME, n_classes=2)
    
    # 2. Load saved weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("❌ Error: No saved model found!")
        print("Please run 'python scripts/train.py' first to save the model.")
        return

    model.to(device)
    model.eval()
    print("✅ System ready. Waiting for news...")
    print("-" * 50)

    while True:
        # User Input
        user_text = input("\n📰 Enter a headline (or 'q' to quit): \n> ")
        
        if user_text.lower() in ['q', 'quit', 'exit']:
            print("Shutting down system.")
            break
            
        if len(user_text) < 5:
            print("⚠️ Text is too short.")
            continue

        # Vorhersage
        encoding = tokenizer.encode_plus(
            user_text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            # Softmax for probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Wir nehmen an: Index 1 = UP, Index 0 = DOWN
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()

        # Output
        if prediction == 1:
            print(f"\n📈 SIGNAL: BUY (UP)")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   AI Interpretation: This headline is bullish.")
        else:
            print(f"\n📉 SIGNAL: SELL / CASH (DOWN)")
            print(f"   Confidence: {confidence:.2%}")
            print(f"   AI Interpretation: Risk detected. The market may fall.")
        print("-" * 50)

if __name__ == "__main__":
    run_live_prediction()
