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

    original_count = len(ds)
    filtered = False

    # 1. Filter for Phishing Type
    if "type" in ds.column_names:
        ds = ds.filter(lambda x: x["type"] == "phishing")
        filtered = True
    elif "label" in ds.column_names:
        ds = ds.filter(lambda x: x["label"] == 1)
        filtered = True

    # 2. Filter for URLs (Remove Emails/Sentences)
    # Heuristic: URLs rarely have spaces. Emails always do.
    text_col = find_text_column(ds, tcfg.text_column_candidates)
    if text_col:
        log("Filtering for URLs (removing text with spaces)...")
        before_url_filter = len(ds)

        ds = ds.filter(lambda x: isinstance(x[text_col], str) and " " not in x[text_col].strip())

        log(f"Removed {before_url_filter - len(ds)} non-URL items.")

    if filtered:
        log(f"Dataset Ready: {len(ds)} Phishing URLs (Filtered out {original_count - len(ds)} items)")
    else:
        log("[WARN] No label column found. Training on MIXED data.")

    # 3. Tokenization
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {ds.column_names}")

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
