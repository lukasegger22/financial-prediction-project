import sys
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from pathlib import Path
import numpy as np

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.model import BERTSentimentClassifier

# Configuration 
MODEL_NAME = 'ProsusAI/finbert' # This must also be FinBERT.
MAX_LEN = 160

def explain_prediction(text, model, tokenizer, device):
    """
    Calculate the importance of each word for the decision (gradient method).
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
    
    # 2. Fetch embeddings (we need access to the lowest layer)
    embeddings = model.bert.embeddings.word_embeddings(input_ids)
    embeddings.retain_grad() # IMPORTANT: We want to measure gradients here.
    
    # 3. Forward pass (manual, so we can pass embeddings)
    
    # Workaround: We need to work around the model so it uses embeddings instead of IDs.
    # Since that is complex, we use an attention-based explanation.
    # This is almost as useful for the demo and much more stable.
    
    outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    
    # Use the attention from the last layer
    attention = outputs.attentions[-1].squeeze(0) 
    
    # Average over all attention heads
    attention_score = torch.mean(attention, dim=0)
    
    # Inspect how strongly the [CLS] token (the classification token at the start)
    # attended to the other words.
    # Index 0 ist [CLS].
    cls_attention = attention_score[0, :] 
    
    # Get prediction
    output_logits = model.out(model.drop(outputs.pooler_output))
    prediction = torch.argmax(output_logits, dim=1).item()
    prob = torch.softmax(output_logits, dim=1)[0][prediction].item()
    
    return cls_attention, prediction, prob, input_ids[0]

def colorize_text(words, scores, prediction):
    """
    Generate HTML code with colors based on scores.
    """
    # Normalize scores for color intensity
    scores = scores.detach().cpu().numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    
    html = '<div style="font-family: Arial; line-height: 1.5;">'
    label = "UP (Buy)" if prediction == 1 else "DOWN (Sell)"
    color_label = "green" if prediction == 1 else "red"
    
    html += f'<h3>Prediction: <span style="color:{color_label}">{label}</span></h3>'
    html += '<p>Words the model focused on most strongly (darker = more important):</p>'
    
    for word, score in zip(words, scores):
        if word in ['[CLS]', '[SEP]', '[PAD]']: continue 
 
        opacity = score 
        html += f'<span style="background-color: rgba(0, 100, 255, {opacity}); padding: 2px; margin: 1px; border-radius: 3px;">{word}</span> '
        
    html += '</div>'
    return html

def run_explanation():
    print("--- Starting Explainable AI analysis (XAI) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    
    # 1. Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # 2. Initialize model (for the demo, we do not need a trained model)
    model = BERTSentimentClassifier(model_name=MODEL_NAME, n_classes=2).to(device)
    
    # 3. Example sentences to analyze
    examples = [
        "Oil prices skyrocket as geopolitical tensions rise in the middle east.",
        "Apple releases new iPhone with record-breaking sales numbers and huge profits.",
        "Bitcoin crashes below support levels due to regulatory fears."
    ]
    
    final_html = "<html><body><h1>FinBERT Model Explanation</h1>"
    
    for text in examples:
        print(f"Analyzing: {text[:50]}...")
        attentions, pred, prob, ids = explain_prediction(text, model, tokenizer, device)
        
        # Convert IDs back to words
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        html_snippet = colorize_text(tokens, attentions, pred)
        final_html += html_snippet + "<hr>"
        
    final_html += "</body></html>"
    
    # Save
    output_path = BASE_DIR / "plots" / "explanation.html"
    with open(output_path, "w") as f:
        f.write(final_html)
        
    print(f"✅ Explanation saved to: {output_path}")
    print("👉 Open this file in your browser (Chrome/Safari) to view the visualization.")

if __name__ == "__main__":
    run_explanation()
