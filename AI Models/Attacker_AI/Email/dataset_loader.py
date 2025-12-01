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

    # --- PHISHING FILTER LOGIC ---
    # Mitake/PhishingURLsANDBenignURLs has a 'label' column.
    # 1 = Phishing, 0 = Benign. We want ONLY Phishing (1).
    if "label" in ds.column_names:
        original_count = len(ds)
        log(f"Filtering dataset for Phishing URLs (label=1)...")
        ds = ds.filter(lambda x: x["label"] == 1)

        new_count = len(ds)
        log(f"Dataset filtered. Keeping {new_count} Phishing URLs (Dropped {original_count - new_count} safe URLs)")
    else:
        log("[WARN] 'label' column not found! Training on MIXED data.")
    # -----------------------------

    text_col = find_text_column(ds, tcfg.text_column_candidates)
    if text_col is None:
        raise ValueError(f"No URL column found. Columns: {ds.column_names}")

    log(f"Using URL column: {text_col}")

    log(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        log("Added special <pad> token to tokenizer.")

    def preprocess(examples):
        urls = examples[text_col]
        urls = [u if isinstance(u, str) else "" for u in urls]
        enc = tokenizer(urls,
                        truncation=True,
                        padding="max_length",
                        max_length=cfg.max_length)
        return enc

    log(f"Tokenizing {len(ds)} examples...")
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)

    return TensorDataset(input_ids, attention_mask), tokenizer
