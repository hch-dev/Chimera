import pandas as pd
import torch
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding
)

# --- 1. SETUP & LOGGING ---
if not os.path.exists('./logs'):
    os.makedirs('./logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION ---
MODEL_NAME = "roberta-base"
DATA_PATH = "./data/phishing_data.csv"
OUTPUT_DIR = "./models/v3_phishing_roberta"

# --- 3. DATA PREPARATION ---
class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # The crash happened here previously because labels were strings.
        # Now they are guaranteed to be Integers.
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # Force Long Tensor
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    logger.info(f"METRICS -- Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}")
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def load_data():
    logger.info(f"Loading dataset from {DATA_PATH}...")
    try:
        # low_memory=False helps with mixed types
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except FileNotFoundError:
        logger.error(f"CRITICAL: Data file not found at {DATA_PATH}. Please add it.")
        exit()

    # --- THE FIX STARTS HERE ---
    
    # 1. Force Text column to string
    if 'text' in df.columns:
        df['text'] = df['text'].astype(str)
    
    # 2. Force Label column to Numeric
    # 'errors="coerce"' turns text like "spam" into NaN (Empty) so we can drop it
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    # 3. Drop any rows that are empty or failed conversion
    initial_len = len(df)
    df = df.dropna(subset=['text', 'label'])
    dropped_rows = initial_len - len(df)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} bad rows (empty or text labels).")

    # 4. Final conversion to Integer (Crucial for PyTorch)
    df['label'] = df['label'].astype(int)
    
    # Debug print to prove it works
    logger.info(f"Unique labels found: {df['label'].unique()}") # Should only see [1 0]
    
    # --- THE FIX ENDS HERE ---
    
    # Split Data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2
    )
    
    logger.info("Tokenizing data...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = PhishingDataset(train_encodings, train_labels)
    val_dataset = PhishingDataset(val_encodings, val_labels)
    
    return train_dataset, val_dataset, tokenizer

# --- 4. MAIN TRAINING LOOP ---
def train():
    logger.info("Initializing Version 3 Training Pipeline...")
    train_dataset, val_dataset, tokenizer = load_data()
    
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=500,        # Reduced logging to keep terminal clean
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        dataloader_pin_memory=False # Fixes the warning for CPU users
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    logger.info("Starting Fine-Tuning...")
    trainer.train()
    
    logger.info(f"Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("TRAINING COMPLETE. Model saved.")

if __name__ == "__main__":
    train()