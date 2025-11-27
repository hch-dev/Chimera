# dataset_loader.py
"""
Dataset loader for the Attacker AI (Generator).
Supports multiple input formats:
    - .txt   (one prompt per line)
    - .json  {"text": "..."}
    - .jsonl one JSON per line
    - .csv   with a 'text' column

Tokenizes with the HF tokenizer passed from the trainer.
"""

import os
import json
import csv
import torch
from torch.utils.data import Dataset


class AttackerDataset(Dataset):
    """
    Loads real-world phishing or malicious-like prompts (safe versions)
    for training the generator model inside Chimera's Attacker Engine.

    This dataset is NOT malicious — it only contains safe examples
    that the generator will learn structure from.
    """

    def __init__(self, dataset_path: str, tokenizer, max_length: int = 128):
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist → {self.dataset_path}")

        # load depending on file type
        ext = os.path.splitext(self.dataset_path)[1].lower()

        if ext == ".txt":
            self.samples = self._load_txt()

        elif ext == ".json":
            self.samples = self._load_json()

        elif ext == ".jsonl":
            self.samples = self._load_jsonl()

        elif ext == ".csv":
            self.samples = self._load_csv()

        else:
            raise ValueError(f"Unsupported dataset extension: {ext}")

        if len(self.samples) == 0:
            raise ValueError("Dataset is empty — found 0 valid samples.")

    # ========================================
    #   LOADERS
    # ========================================

    def _load_txt(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        return lines

    def _load_json(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "text" in data:
            return [data["text"]]
        elif isinstance(data, list):
            return [x["text"] for x in data if "text" in x]
        else:
            raise ValueError("JSON must contain a 'text' field or a list of such objects.")

    def _load_jsonl(self):
        samples = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "text" in obj:
                    samples.append(obj["text"])
        return samples

    def _load_csv(self):
        samples = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "text" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'text' column")

            for row in reader:
                if row["text"].strip():
                    samples.append(row["text"].strip())
        return samples

    # ========================================
    #   TOKENIZATION
    # ========================================

    def _encode(self, text: str):
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # shape: [1, seq_len] → we want [seq_len]
        return encoding["input_ids"].squeeze(0)

    # ========================================
    #   PYTORCH DATASET API
    # ========================================

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self._encode(text)
        return ids

if __name__ == "__main__":
    ds = AttackerDataset("attacker_data/train.jsonl")
    print("Samples:", len(ds))
    print("First sample:", ds[0])
