import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from generator_model import GeneratorModel
from configs import GeneratorConfig

def load_tokenizer():
    print("[LOG] Using GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # CRITICAL: Match Training (vocab size 50258)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    return tokenizer

@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=64, temperature=0.8):
    model.eval()

    # 1. Start with "Subject:"
    start_text = "Subject:"
    input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)

    # 2. Manual Autoregressive Loop
    curr_ids = input_ids

    # We loop up to max_new_tokens, but we MUST stop if we hit model.max_length
    for _ in range(max_new_tokens):

        # --- CRITICAL FIX: Stop if we are at the model's limit ---
        if curr_ids.size(1) >= model.max_length:
            break
        # ---------------------------------------------------------

        # Forward pass
        logits = model(curr_ids) # Shape: (1, seq_len, vocab_size)

        # Get logits for the last token only
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        next_token_logits = next_token_logits / temperature

        # Sample the next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # Shape: (1, 1)

        # Append to sequence
        curr_ids = torch.cat([curr_ids, next_token], dim=1)

        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(curr_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="models/generator_epoch1.pt")
    args = parser.parse_args()

    device = torch.device("cpu")

    # 1. Load Tokenizer
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)
    print(f"[LOG] Vocab size: {vocab_size}")

    # 2. Build Model
    config = GeneratorConfig()
    model = GeneratorModel(vocab_size=vocab_size, d_model=config.d_model,
                          nhead=config.nhead, num_layers=config.num_layers,
                          max_length=config.max_length)

    # 3. Load Weights
    if not os.path.exists(args.ckpt):
        print(f"Error: Model not found at {args.ckpt}.")
        return

    print(f"[LOG] Loading model from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)

    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    print("\n=== GENERATED PHISHING EMAILS ===")
    for i in range(5):
        email = generate_samples(model, tokenizer, device)
        print(f"\n--- Email {i+1} ---\n{email}\n")

if __name__ == "__main__":
    main()
