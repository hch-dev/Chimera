# dataset_loader.py
"""
Loads RonakAJ/phising_email (Hugging Face) or any local dataset with
a text column. Returns a tokenized torch TensorDataset and a tokenizer.
"""

from datasets import load_dataset
from torch.utils.data import TensorDataset
import torch
from transformers import AutoTokenizer
from configs import GeneratorConfig, TrainingConfig
from utils import log

def find_text_column(ds, candidates):
    # ds is a datasets.Dataset (single split) or DatasetDict
    for c in candidates:
        if c in ds.column_names:
            return c
    # fallback: choose first string column
    for c in ds.column_names:
        # we only check first row type
        val = ds[c][0]
        if isinstance(val, str):
            return c
    return None

def load_and_tokenize(cfg: GeneratorConfig, tcfg: TrainingConfig):
    log(f"Loading dataset: {tcfg.dataset_name}")
    ds = load_dataset(tcfg.dataset_name, split="train")  # RonakAJ/phising_email single split
    text_col = find_text_column(ds, tcfg.text_column_candidates)
    if text_col is None:
        raise ValueError(f"No text column found in dataset. Columns: {ds.column_names}")
    log(f"Found text column: {text_col} (samples: {len(ds)})")

    log(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        texts = examples[text_col]
        # basic cleaning: ensure strings
        texts = [t if isinstance(t, str) else "" for t in texts]
        enc = tokenizer(texts,
                        truncation=True,
                        padding="max_length",
                        max_length=cfg.max_length)
        return enc

    log(f"Tokenizing {len(ds)} examples (max_length={cfg.max_length})...")
    tokenized = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)

    dataset = TensorDataset(input_ids, attention_mask)
    log(f"Tokenized shape: {input_ids.shape}")
    return dataset, tokenizer
