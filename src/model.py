import torch.nn as nn
from transformers import BertModel

class BERTSentimentClassifier(nn.Module):
    """
    Zentrale Modell-Definition.
    Verhindert Code-Duplizierung in train.py und backtest.py.
    """
    def __init__(self, model_name, n_classes):
        super(BERTSentimentClassifier, self).__init__()
        # Wir laden das Basis-Modell
        self.bert = BertModel.from_pretrained(model_name)
        
        # Dropout gegen Overfitting
        self.drop = nn.Dropout(p=0.3)
        
        # Der finale Layer (Entscheidungsschicht)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = output.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)