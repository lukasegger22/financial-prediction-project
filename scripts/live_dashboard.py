import sys
import torch
import torch.nn.functional as F
import feedparser
from transformers import BertTokenizer
from pathlib import Path
from datetime import datetime

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.model import BERTSentimentClassifier

# --- CONFIGURATION ---
MODEL_NAME = 'ProsusAI/finbert' 
MODEL_PATH = BASE_DIR / "models" / "finbert_trained.pth"
MAX_LEN = 160
RSS_URL = "https://finance.yahoo.com/news/rssindex"
OUTPUT_HTML = BASE_DIR / "plots" / "live_dashboard.html"

def get_live_news(limit=7):
    """Fetch the latest headlines from Yahoo Finance."""
    print(f"📡 Connecting to Yahoo Finance News Feed...")
    feed = feedparser.parse(RSS_URL)
    headlines = []
    for entry in feed.entries[:limit]:
        headlines.append({'title': entry.title})
    return headlines

def generate_html(results, summary):
    """Create a professional HTML view for screenshots."""
    
    # CSS styling for the Bloomberg Terminal look
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Market Sentiment Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f6f9; color: #333; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            .header {{ border-bottom: 2px solid #eaeaea; padding-bottom: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }}
            .title {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .timestamp {{ color: #7f8c8d; font-size: 14px; }}
            
            .summary-box {{ padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 30px; color: white; font-weight: bold; font-size: 18px; }}
            .bullish {{ background: linear-gradient(135deg, #2ecc71, #27ae60); }}
            .bearish {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
            .neutral {{ background: linear-gradient(135deg, #95a5a6, #7f8c8d); }}
            
            table {{ width: 100%; border-collapse: collapse; }}
            th {{ text-align: left; padding: 12px; border-bottom: 2px solid #ddd; color: #7f8c8d; font-size: 12px; text-transform: uppercase; }}
            td {{ padding: 15px 12px; border-bottom: 1px solid #eee; font-size: 14px; }}
            .tag {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; color: white; display: inline-block; width: 60px; text-align: center; }}
            .tag-up {{ background-color: #27ae60; }}
            .tag-down {{ background-color: #c0392b; }}
            .headline {{ font-weight: 500; }}
            .conf-bar {{ height: 4px; background: #eee; width: 100px; margin-top: 5px; border-radius: 2px; }}
            .conf-fill {{ height: 100%; border-radius: 2px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">🤖 AI Market Sentiment Dashboard</div>
                <div class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            
            <div class="summary-box {summary['class']}">
                MARKET TENDENCY: {summary['text']} 
                <div style="font-size: 14px; margin-top: 5px; opacity: 0.9;">
                    {summary['bullish_count']} Positive vs {summary['bearish_count']} Negative Signals
                </div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Live Headline (Source: Yahoo Finance)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for res in results:
        tag_class = "tag-up" if res['prediction'] == 1 else "tag-down"
        tag_text = "UP" if res['prediction'] == 1 else "DOWN"
        bar_color = "#27ae60" if res['prediction'] == 1 else "#c0392b"
        
        html += f"""
        <tr>
            <td><span class="tag {tag_class}">{tag_text}</span></td>
            <td>
                {res['confidence']:.1%}
                <div class="conf-bar"><div class="conf-fill" style="width: {res['confidence']*100}%; background-color: {bar_color};"></div></div>
            </td>
            <td class="headline">{res['text']}</td>
        </tr>
        """
        
    html += """
                </tbody>
            </table>
            <div style="margin-top: 20px; text-align: center; font-size: 12px; color: #aaa;">
                Powered by FinBERT & PyTorch • Live Inference System
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(OUTPUT_HTML, "w", encoding='utf-8') as f:
        f.write(html)
    print(f"\n✅ HTML dashboard saved to: {OUTPUT_HTML}")
    print("👉 Open this file in your browser for the screenshot.")

def run_dashboard():
    print(f"🔴 LIVE AI MARKET SENTIMENT DASHBOARD ({datetime.now().strftime('%H:%M:%S')})")
    print("-" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    print("🧠 Loading FinBERT AI model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BERTSentimentClassifier(model_name=MODEL_NAME, n_classes=2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print("❌ Error: Model not found.")
        return

    model.to(device)
    model.eval()

    news_items = get_live_news(limit=7)
    if not news_items: return

    results = []
    bullish_count = 0
    bearish_count = 0

    for item in news_items:
        text = item['title']
        encoding = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=MAX_LEN,
            return_token_type_ids=False, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
            
        results.append({
            'text': text,
            'prediction': prediction,
            'confidence': confidence
        })
        
        if prediction == 1: bullish_count += 1
        else: bearish_count += 1
        
        # Terminal Output
        icon = "🟢" if prediction == 1 else "🔴"
        print(f"{icon} {confidence:.1%} | {text[:60]}...")

    # Summary Logic
    if bullish_count > bearish_count:
        sum_text = "BULLISH (BUY)"
        sum_class = "bullish"
    elif bearish_count > bullish_count:
        sum_text = "BEARISH (SELL)"
        sum_class = "bearish"
    else:
        sum_text = "NEUTRAL"
        sum_class = "neutral"
        
    summary = {
        'text': sum_text,
        'class': sum_class,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count
    }
    
    generate_html(results, summary)

if __name__ == "__main__":
    run_dashboard()
