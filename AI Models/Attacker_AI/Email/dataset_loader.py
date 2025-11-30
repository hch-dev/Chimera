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
    for c in ds.column_names:
        if isinstance(ds[c][0], str):
            return c
    return None

def load_and_tokenize(cfg: GeneratorConfig, tcfg: TrainingConfig):
    log(f"Loading dataset: {tcfg.dataset_name}")
    ds = load_dataset(tcfg.dataset_name, split="train")
    text_col = find_text_column(ds, tcfg.text_column_candidates)
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {ds.column_names}")
    log(f"Found text column: {text_col}")

    log(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # CRITICAL FIX: Add a real pad token so EOS is not ignored
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        log("Added special <pad> token to tokenizer.")

    def preprocess(examples):
        texts = examples[text_col]
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
