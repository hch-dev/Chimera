import os
import argparse
import torch
from transformers import AutoTokenizer
from generator_model import GeneratorModel
from configs import GeneratorConfig

def load_tokenizer():
    print("[LOG] Using GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})

    return tokenizer

@torch.no_grad()
def generate_samples(model, tokenizer, device, max_new_tokens=128, temperature=0.8):
    model.eval()

    # Start with "Subject:" to guide email generation
    start_text = "Subject: Urgent Action Required"
    input_ids = tokenizer.encode(start_text, return_tensors='pt').to(device)

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
    # Default to models_email folder
    default_ckpt = "models_email/generator_final.pt"
    if os.path.exists("models_email/generator_interrupted.pt"):
        default_ckpt = "models_email/generator_interrupted.pt"

    parser.add_argument("--ckpt", type=str, default=default_ckpt)
    args = parser.parse_args()

    device = torch.device("cpu")

    tokenizer = load_tokenizer()
    gcfg = GeneratorConfig()
    model = GeneratorModel(max_length=gcfg.max_length)
    model.gpt.resize_token_embeddings(len(tokenizer))

    if os.path.exists(args.ckpt):
        print(f"[LOG] Loading weights from {args.ckpt}...")
        checkpoint = torch.load(args.ckpt, map_location=device)

        if "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint

        model.load_state_dict(state)
    else:
        print(f"[WARN] No checkpoint found at {args.ckpt}. Using raw GPT-2.")

    model.to(device)

    print("\n=== GENERATED PHISHING EMAILS ===")
    for i in range(5):
        email = generate_samples(model, tokenizer, device)
        print(f"\n--- Email {i+1} ---\n{email}\n")

if __name__ == "__main__":
    main()
