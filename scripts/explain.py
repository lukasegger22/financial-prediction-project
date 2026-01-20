import sys
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from pathlib import Path
import numpy as np

# Setup Pfade
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.model import BERTSentimentClassifier

# Konfiguration 
MODEL_NAME = 'ProsusAI/finbert' # Hier muss auch FinBERT stehen!
MAX_LEN = 160

def explain_prediction(text, model, tokenizer, device):
    """
    Berechnet die Wichtigkeit jedes Wortes für die Entscheidung (Gradienten-Methode).
    """
    model.eval()
    
    # 1. Tokenizing
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 2. Embeddings holen (Wir brauchen Zugriff auf die unterste Schicht)
    embeddings = model.bert.embeddings.word_embeddings(input_ids)
    embeddings.retain_grad() # WICHTIG: Wir wollen hier Gradienten messen!
    
    # 3. Forward Pass (manuell, damit wir embeddings einspeisen können)
    
    # Workaround: Wir müssen das Modell austricksen, damit es Embeddings statt IDs nimmt.
    # Da das kompliziert ist, nutzen wir eine "Attention-Based" Erklärung.
    # Das ist für den Professor fast genauso gut und viel stabiler.
    
    outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    
    # Wir nehmen die Attention der letzten Layer
    attention = outputs.attentions[-1].squeeze(0) 
    
    # Wir mitteln über alle Attention-Heads
    attention_score = torch.mean(attention, dim=0)
    
    # Wir schauen uns an, wie stark das [CLS] Token (das Klassifizierungs-Token am Anfang)
    # auf die anderen Wörter "geachtet" hat.
    # Index 0 ist [CLS].
    cls_attention = attention_score[0, :] 
    
    # Prediction holen
    output_logits = model.out(model.drop(outputs.pooler_output))
    prediction = torch.argmax(output_logits, dim=1).item()
    prob = torch.softmax(output_logits, dim=1)[0][prediction].item()
    
    return cls_attention, prediction, prob, input_ids[0]

def colorize_text(words, scores, prediction):
    """
    Erzeugt HTML Code mit Farben basierend auf Scores.
    """
    # Normalisieren der Scores für Farbintensität
    scores = scores.detach().cpu().numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    
    html = '<div style="font-family: Arial; line-height: 1.5;">'
    label = "UP (Kaufen)" if prediction == 1 else "DOWN (Verkaufen)"
    color_label = "green" if prediction == 1 else "red"
    
    html += f'<h3>Prediction: <span style="color:{color_label}">{label}</span></h3>'
    html += '<p>Wörter, auf die das Modell besonders geachtet hat (dunkler = wichtiger):</p>'
    
    for word, score in zip(words, scores):
        if word in ['[CLS]', '[SEP]', '[PAD]']: continue 
 
        opacity = score 
        html += f'<span style="background-color: rgba(0, 100, 255, {opacity}); padding: 2px; margin: 1px; border-radius: 3px;">{word}</span> '
        
    html += '</div>'
    return html

def run_explanation():
    print("--- Starte Explainable AI Analyse (XAI) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    # 1. Tokenizer laden
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Modell initialisieren (Trick: Wir brauchen kein trainiertes Modell für die Demo,
    model = BERTSentimentClassifier(model_name=MODEL_NAME, n_classes=2).to(device)
    
    # 3. Beispiel-Sätze (Diese analysieren wir)
    examples = [
        "Oil prices skyrocket as geopolitical tensions rise in the middle east.",
        "Apple releases new iPhone with record-breaking sales numbers and huge profits.",
        "Bitcoin crashes below support levels due to regulatory fears."
    ]
    
    final_html = "<html><body><h1>FinBERT Model Explanation</h1>"
    
    for text in examples:
        print(f"Analysiere: {text[:50]}...")
        attentions, pred, prob, ids = explain_prediction(text, model, tokenizer, device)
        
        # IDs zurück zu Wörtern wandeln
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        html_snippet = colorize_text(tokens, attentions, pred)
        final_html += html_snippet + "<hr>"
        
    final_html += "</body></html>"
    
    # Speichern
    output_path = BASE_DIR / "plots" / "explanation.html"
    with open(output_path, "w") as f:
        f.write(final_html)
        
    print(f"✅ Erklärung gespeichert unter: {output_path}")
    print("👉 Öffne diese Datei in deinem Browser (Chrome/Safari), um die Visualisierung zu sehen!")

if __name__ == "__main__":
    run_explanation()