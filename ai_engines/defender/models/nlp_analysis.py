import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
class NLPPhishingDetector(nn.Module):
    """
    NLP Model for phishing email detection.
    Produces:
        - Classification score (phishing probability)
        - Embedding for fusion model
    """
    def __init__(self, embedding_dim=256):
        super().__init__()
        # Step 1: Load pretrained tokenizer + DistilBERT encoder
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        bert_output_dim = self.bert.config.hidden_size  # usually 768
        # Step 2: Projection head to reduce dimensionality for fusion model
        self.embedding_head = nn.Sequential(
            nn.Linear(bert_output_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Step 3: Classification head (binary)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # probability output
        )
    # ---------------------------------------------------
    # TOKENIZATION & FORWARD PASS
    # ---------------------------------------------------
    def tokenize(self, text, max_len=256):
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
    def forward(self, text, return_embedding=False):
        """
        text: raw email content
        return_embedding: set True when combining with CNN/GNN for the final model
        """
        inputs = self.tokenize(text)
        outputs = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        # Take [CLS] or mean pooling
        pooled = outputs.last_hidden_state[:, 0, :]
        # Embedding vector for fusion model
        embedding = self.embedding_head(pooled)
        if return_embedding:
            return embedding
        # Phishing probability
        prob = self.classifier(embedding)
        return prob
    # ---------------------------------------------------
    # SAVE / LOAD UTILITIES
    # ---------------------------------------------------
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))
        return self
