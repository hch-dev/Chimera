# generator_inference.py
import os
import argparse
import torch
from transformers import AutoTokenizer
from generator_model import GeneratorModel
from configs import GeneratorConfig

def load_tokenizer():
    print("[LOG] Using GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # MATCH TRAINING LOGIC: Add PAD token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    return tokenizer

@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=64, temperature=0.8):
    model.eval()

    start_text = "http"
    input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)

    # Use HuggingFace's generate
    output_ids = model.gpt.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    # Default to the correct folder (models_url)
    parser.add_argument("--ckpt", type=str, default="models_url/generator_final.pt")
    args = parser.parse_args()

    device = torch.device("cpu")

    # 1. Load Tokenizer
    tokenizer = load_tokenizer()

    # 2. Build Model
    gcfg = GeneratorConfig()
    model = GeneratorModel(max_length=gcfg.max_length)

    # 3. Resize embeddings to match training (CRITICAL FIX)
    model.gpt.resize_token_embeddings(len(tokenizer))

    # 4. Load Weights
    if os.path.exists(args.ckpt):
        print(f"[LOG] Loading fine-tuned weights from {args.ckpt}...")
        checkpoint = torch.load(args.ckpt, map_location=device)

        # Handle different save formats (state_dict vs full checkpoint)
        state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state)
    else:
        print(f"[WARN] No checkpoint found at {args.ckpt}. Using raw GPT-2.")

    model.to(device)

    print("\n=== GENERATED PHISHING URLs ===")
    for i in range(10):
        url = generate_samples(model, tokenizer, device)
        print(f"{i+1}. {url}")

if __name__ == "__main__":
    main()
