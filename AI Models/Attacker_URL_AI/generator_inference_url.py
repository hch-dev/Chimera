# generator_inference_url.py

import os
import torch
from transformers import AutoTokenizer
from generator_model_url import GeneratorURLModel
from configs_url import GeneratorURLConfig
import re


# ---------------------------
# Clean URL to enforce format
# ---------------------------
def fix_url(url: str) -> str:
    if not isinstance(url, str):
        return ""

    # Basic cleanup
    url = url.strip().lower()
    url = re.sub(r"[^a-z0-9\-._/]", "", url)

    # Remove subword repeats like "ravityravityravity"
    url = re.sub(r"(.{3,10})\1{1,}", r"\1", url)

    # Extract meaningful sequences (letters/numbers only)
    words = re.findall(r"[a-z0-9]+", url)

    # If model produced nothing useful, fallback to safe base
    if not words:
        words = ["secure", "login"]

    # Pick top 2–3 meaningful chunks
    parts = []
    for w in words:
        if len(w) > 3 and len(w) < 20 and not any(w in p for p in parts):
            parts.append(w)
        if len(parts) >= 3:
            break

    if not parts:
        parts = ["account", "verify"]

    # Join chunks to form realistic domain
    domain = "-".join(parts)

    # Enforce final URL structure
    final = f"https://www.{domain}.com"

    return final


# ---------------------------
# Generate from model
# ---------------------------
def generate_url(model, tokenizer, max_length=64, device="cpu"):
    model.eval()

    # Start with BOS token
    input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)  # (1,1)

    generated_ids = []

    with torch.no_grad():
        for _ in range(max_length):

            # Forward pass -> logits
            logits = model(input_ids)            # (1, seq_len, vocab)
            logits = logits[:, -1, :]            # (1, vocab)

            # Top-k sampling
            top_k = 50
            vals, idxs = torch.topk(logits, top_k, dim=-1)  # both shape: (1, top_k)
            probs = torch.softmax(vals, dim=-1)

            # Sample 1 token (returns shape (1,1))
            sampled_idx = torch.multinomial(probs, num_samples=1)       # (1,1)
            next_token_id = idxs.gather(-1, sampled_idx)                # (1,1)

            # Convert to scalar int
            token_id = next_token_id.item()
            generated_ids.append(token_id)

            # Append token to running input_ids
            input_ids = torch.cat([input_ids, next_token_id], dim=1)    # shapes align: (1,n) + (1,1)

            # Decode partial output
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Stop on EOS
            if token_id == tokenizer.eos_token_id:
                break

            # Stop once URL clearly finished
            if decoded.endswith(".com"):
                break

            # Stop if repeating (AAA AAA)
            if len(generated_ids) > 6:
                if generated_ids[-3:] == generated_ids[-6:-3]:
                    break

    # Final clean URL
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return fix_url(text)


# ---------------------------
# Main
# ---------------------------
def main():
    print("[DEBUG] Starting URL generator…")

    gcfg = GeneratorURLConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] Using device:", device)

    # ---------------------------
    # Load Tokenizer
    # ---------------------------
    print("[DEBUG] Loading GPT-2 tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    vocab_size = len(tokenizer)
    print("[DEBUG] Tokenizer loaded. Vocab size =", vocab_size)

    # ---------------------------
    # Build Generator Model
    # ---------------------------
    print("[DEBUG] Building GeneratorURLModel…")
    model = GeneratorURLModel(
        vocab_size=vocab_size,
        d_model=gcfg.d_model,
        nhead=gcfg.nhead,
        num_layers=gcfg.num_layers,
        max_length=gcfg.max_length,
        dropout=gcfg.dropout,
    ).to(device)

    # ---------------------------
    # Build LOCAL & PORTABLE checkpoint path
    # ---------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(BASE_DIR, "models_url", "generator_url_epoch3.pt")

    print("[DEBUG] Loading checkpoint:", ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("[DEBUG] Checkpoint loaded successfully.")

    # ---------------------------
    # Generate a URL
    # ---------------------------
    print("[DEBUG] Generating URL…")
    url = generate_url(model, tokenizer, gcfg.max_length, device)

    print("\n====================================")
    print("Generated URL:")
    print(url)
    print("====================================\n")


if __name__ == "__main__":
    main()
