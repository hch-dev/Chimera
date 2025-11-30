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

    # MATCH TRAINING LOGIC: Use EOS as PAD instead of adding a new token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=64, temperature=0.8):
    model.eval()

    # Start with a simple "http" prompt to guide the model towards URLs
    start_text = "http"
    input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)

    # Generate
    output_ids = model.transformer.generate(
        input_ids,
        max_length=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    # Default to the folder where configs.py saves models
    parser.add_argument("--ckpt", type=str, default="models_url/generator_final.pt")
    args = parser.parse_args()

    device = torch.device("cpu")

    # 1. Load Tokenizer
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer)

    # 2. Build Model
    config = GeneratorConfig()
    # Ensure config matches training dimensions
    model = GeneratorModel(vocab_size=vocab_size, d_model=config.d_model,
                          nhead=config.nhead, num_layers=config.num_layers,
                          max_length=config.max_length)

    # 3. Load Weights
    if not os.path.exists(args.ckpt):
        print(f"Error: Model not found at {args.ckpt}. Please wait for training to finish.")
        return

    print(f"[LOG] Loading model from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)

    # Handle potential key mismatch if saved differently
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    print("\n=== GENERATED PHISHING URLS ===")
    for i in range(5):
        url = generate_samples(model, tokenizer, device)
        print(f"{i+1}. {url}")

if __name__ == "__main__":
    main()
