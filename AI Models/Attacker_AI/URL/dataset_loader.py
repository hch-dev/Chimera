# dataset_loader.py
import re
from datasets import load_dataset, Dataset
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

def extract_urls_from_text(text):
    """
    Extracts http/https URLs from a block of text.
    """
    if not isinstance(text, str): return []
    # Basic regex to find URLs
    urls = re.findall(r'(https?://[^\s<>"]+|www\.[^\s<>"]+)', text)
    return [u.rstrip('.,;)') for u in urls] # Clean trailing punctuation

def load_and_tokenize(cfg: GeneratorConfig, tcfg: TrainingConfig):
    log(f"Loading dataset: {tcfg.dataset_name}")
    ds = load_dataset(tcfg.dataset_name, split="train")

    original_count = len(ds)
    filtered = False

    # --- FILTER FOR PHISHING EMAILS FIRST ---
    # Dataset uses 'labels' column: 0=Safe, 1=Phishing, 2=... 3=...
    # We typically assume 1, 2, 3 are suspicious/phishing in this dataset.
    if "labels" in ds.column_names:
        log(f"Detected 'labels' column. Filtering for Phishing (label > 0)...")
        ds = ds.filter(lambda x: x["labels"] > 0)
        filtered = True
    elif "label" in ds.column_names:
        ds = ds.filter(lambda x: x["label"] == 1)
        filtered = True

    if filtered:
        log(f"Found {len(ds)} Phishing Emails. Now extracting URLs...")

    # --- EXTRACT URLs FROM EMAILS ---
    text_col = find_text_column(ds, tcfg.text_column_candidates)
    if not text_col:
        raise ValueError(f"No text column found. Available: {ds.column_names}")

    # Extract all URLs into a list
    extracted_urls = []
    for row in ds:
        content = row[text_col]
        urls = extract_urls_from_text(content)
        extracted_urls.extend(urls)

    # Create a new dataset purely of URLs
    # Limit to 20k to prevent memory overflow if millions are found
    if len(extracted_urls) > 50000:
        extracted_urls = extracted_urls[:50000]

    log(f"Extracted {len(extracted_urls)} unique phishing URLs from emails.")

    # If no URLs found, fallback (safety check)
    if len(extracted_urls) == 0:
        raise ValueError("No URLs could be extracted from the phishing emails! Dataset might be clean.")

    # --- TOKENIZATION ---
    log(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        log("Added special <pad> token to tokenizer.")

    # Tokenize the list of strings directly
    encodings = tokenizer(extracted_urls, truncation=True, padding="max_length", max_length=cfg.max_length)

    input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)

    return TensorDataset(input_ids, attention_mask), tokenizer
