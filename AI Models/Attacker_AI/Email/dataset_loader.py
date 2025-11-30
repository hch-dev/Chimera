# dataset_loader.py
from datasets import load_dataset
from torch.utils.data import TensorDataset
import torch
from transformers import AutoTokenizer
from configs import GeneratorConfig, TrainingConfig
from utils import log

def find_text_column(ds, candidates):
    for c in candidates:
        if c in ds.column_names:
            return c
    return None

def load_and_tokenize(cfg: GeneratorConfig, tcfg: TrainingConfig):
    log(f"Loading dataset: {tcfg.dataset_name}")
    ds = load_dataset(tcfg.dataset_name, split="train")

    # --- UNIVERSAL FILTERING LOGIC ---
    original_count = len(ds)
    filtered = False

    # Case 1: renemel/compiled-phishing-dataset (Columns: 'text', 'type')
    if "type" in ds.column_names:
        log(f"Detected 'type' column. Filtering for 'phishing'...")
        ds = ds.filter(lambda x: x["type"] == "phishing")
        filtered = True

    # Case 2: RonakAJ/phising_email (Columns: 'Email Text', 'Email Type')
    elif "Email Type" in ds.column_names:
        log(f"Detected 'Email Type' column. Filtering for 'Phishing Email'...")
        ds = ds.filter(lambda x: x["Email Type"] == "Phishing Email")
        filtered = True

    # Case 3: label column (Common standard, usually 1=Phishing)
    elif "label" in ds.column_names:
        # Check if it's an integer (1) or string ('phishing')
        sample_val = ds[0]["label"]
        if isinstance(sample_val, int):
            log(f"Detected 'label' column (int). Assuming 1 = Phishing...")
            ds = ds.filter(lambda x: x["label"] == 1)
        elif isinstance(sample_val, str):
             log(f"Detected 'label' column (str). Filtering for 'phishing'...")
             ds = ds.filter(lambda x: x["label"].lower() == "phishing")
        filtered = True

    if filtered:
        new_count = len(ds)
        log(f"Dataset filtered. Keeping {new_count} Phishing examples (Dropped {original_count - new_count} safe emails)")
    else:
        log("[WARN] Could not identify a label column to filter! Training on MIXED data (Attacker might learn to be safe).")
    # ----------------------------------------------

    text_col = find_text_column(ds, tcfg.text_column_candidates)
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {ds.column_names}")

    log(f"Using text column: {text_col}")

    log(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        log("Added special <pad> token to tokenizer.")

    def preprocess(examples):
        texts = examples[text_col]
        # Handle None/Empty values
        texts = [t if isinstance(t, str) else "" for t in texts]
        enc = tokenizer(texts,
                        truncation=True,
                        padding="max_length",
                        max_length=cfg.max_length)
        return enc

    log(f"Tokenizing {len(ds)} examples...")
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)

    return TensorDataset(input_ids, attention_mask), tokenizer
