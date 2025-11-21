import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from ai_engines.defender.models.nlp_model import NLPPhishingClassifier
# -------------------------------------------------------
#  Custom Dataset for NLP Phishing Detection
# -------------------------------------------------------
class PhishingTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        """
        texts: list of raw text samples
        labels: list of 0 or 1
        tokenizer: callable that converts text → list of token IDs
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        token_ids = torch.tensor(self.tokenizer(self.texts[idx]), dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return token_ids, label
# -------------------------------------------------------
#  Collate function for dynamic batching
# -------------------------------------------------------
def collate_batch(batch):
    token_seqs, labels = zip(*batch)
    padded_tokens = pad_sequence(token_seqs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_tokens, labels
# -------------------------------------------------------
#  Dummy tokenizer (replace with your tokenizer later)
# -------------------------------------------------------
def simple_tokenizer(text):
    """
    Converts text to word-index tokens.
    This is a placeholder — replace with:
      - Word2Vec
      - BERT tokenizer
      - Custom tokenizer
      - etc.
    """
    words = text.lower().split()
    vocab = {w: i+1 for i, w in enumerate(set(words))}
    return [vocab[w] for w in words]
# -------------------------------------------------------
#  Training Loop
# -------------------------------------------------------
def train_nlp_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    save_path="nlp_model.pth",
    epochs=5,
    batch_size=16,
    learning_rate=1e-4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    # ------------------------------
    # Load Dataset
    # ------------------------------
    train_dataset = PhishingTextDataset(train_texts, train_labels, simple_tokenizer)
    val_dataset = PhishingTextDataset(val_texts, val_labels, simple_tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    # ------------------------------
    # Load Model
    # ------------------------------
    model = NLPPhishingClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # ------------------------------
    # Training Epochs
    # ------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_tokens, batch_labels in train_loader:
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_tokens).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        # ------------------------------
        # Validation Step
        # ------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_tokens, val_labels in val_loader:
                val_tokens = val_tokens.to(device)
                val_labels = val_labels.to(device)
                outputs = model(val_tokens).squeeze()
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    # ------------------------------
    # Save Model Checkpoint
    # ------------------------------
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
# -------------------------------------------------------
#  Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    # Dummy samples (replace with real dataset later)
    train_texts = [
        "Verify your account immediately by clicking this link",
        "Your Amazon package has shipped",
        "Your bank credentials are required",
        "Meeting scheduled for tomorrow"
    ]
    train_labels = [1, 0, 1, 0]
    val_texts = [
        "Reset your password now",
        "Your order is confirmed"
    ]
    val_labels = [1, 0]
    train_nlp_model(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=3
    )
