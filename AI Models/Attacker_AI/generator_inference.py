#!/usr/bin/env python3
"""
generator_inference.py

Robust inference script for the GeneratorModel in this project.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from generator_model import GeneratorModel
from configs import GeneratorConfig

def load_tokenizer(cfg: GeneratorConfig):
    """
    Use GPT-2 tokenizer as the working tokenizer. Ensure special tokens exist.
    """
    print("[LOG] Using GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    # ensure special tokens exist for safe encoding/decoding
    added = {}
    if tokenizer.pad_token is None:
        added["pad_token"] = "<pad>"
    if tokenizer.bos_token is None:
        added["bos_token"] = "<bos>"
    if tokenizer.eos_token is None:
        added["eos_token"] = "<eos>"

    if added:
        print(f"[LOG] Adding special tokens to tokenizer: {list(added.keys())}")
        tokenizer.add_special_tokens(added)
    return tokenizer

def safe_load_state(model: torch.nn.Module, state: dict):
    """
    Load a state dict into model while handling mismatched vocab sizes:
    - Resize token embedding weight and LM head weight/bias if necessary (randomly pad/truncate).
    - Then load_state_dict with strict=False and report missing/unexpected keys.
    """
    model_state = model.state_dict()
    new_state = dict(state)  # copy so we can mutate

    emb_key = "token_emb.weight"
    if emb_key in state and emb_key in model_state and state[emb_key].shape != model_state[emb_key].shape:
        old = state[emb_key]
        new = model_state[emb_key]
        old_vocab, dim = old.shape
        new_vocab, _ = new.shape
        print(f"[WARN] Resizing token embedding: ckpt_vocab={old_vocab}, model_vocab={new_vocab}")
        if new_vocab > old_vocab:
            pad = torch.randn(new_vocab - old_vocab, dim, device=old.device) * 0.02
            new_state[emb_key] = torch.cat([old, pad], dim=0)
        else:
            new_state[emb_key] = old[:new_vocab]

    if "head.weight" in state and "head.weight" in model_state and state["head.weight"].shape != model_state["head.weight"].shape:
        old = state["head.weight"]
        new = model_state["head.weight"]
        old_vocab, dim = old.shape
        new_vocab, _ = new.shape
        print(f"[WARN] Resizing head.weight: ckpt_vocab={old_vocab}, model_vocab={new_vocab}")
        if new_vocab > old_vocab:
            pad = torch.randn(new_vocab - old_vocab, dim, device=old.device) * 0.02
            new_state["head.weight"] = torch.cat([old, pad], dim=0)
        else:
            new_state["head.weight"] = old[:new_vocab]

    if "head.bias" in state and "head.bias" in model_state and state["head.bias"].shape != model_state["head.bias"].shape:
        old = state["head.bias"]
        new = model_state["head.bias"]
        old_vocab = old.shape[0]
        new_vocab = new.shape[0]
        print(f"[WARN] Resizing head.bias: ckpt_vocab={old_vocab}, model_vocab={new_vocab}")
        if new_vocab > old_vocab:
            pad = torch.zeros(new_vocab - old_vocab, device=old.device)
            new_state["head.bias"] = torch.cat([old, pad], dim=0)
        else:
            new_state["head.bias"] = old[:new_vocab]

    load_result = model.load_state_dict(new_state, strict=False)
    if load_result.missing_keys:
        print("[INFO] Missing keys when loading checkpoint:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("[INFO] Unexpected keys in checkpoint:", load_result.unexpected_keys)
    return model

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    top_k = int(top_k)
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[-1]
        logits[logits < min_values] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

@torch.no_grad()
def generate_samples(model, tokenizer, device, prompt="", max_new_tokens=80,
                     temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
    """
    Generate a single sample conditioned on prompt text (string).
    Handles empty prompts by seeding a safe start token and prevents exceeding model.max_length.
    """
    model.eval()
    if prompt is None:
        prompt = ""

    # encode prompt
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
    input_ids = enc.get("input_ids", None)
    if input_ids is None:
        input_ids = torch.tensor([[tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1]])
    else:
        if input_ids.size(1) == 0:
            # pick a safe start token
            if getattr(tokenizer, "bos_token_id", None) is not None:
                start_id = tokenizer.bos_token_id
            elif getattr(tokenizer, "eos_token_id", None) is not None:
                start_id = tokenizer.eos_token_id
            elif getattr(tokenizer, "pad_token_id", None) is not None:
                start_id = tokenizer.pad_token_id
            else:
                start_id = 1
            input_ids = torch.tensor([[start_id]], dtype=torch.long)

    input_ids = input_ids.to(device)

    if "attention_mask" in enc:
        attention_mask = enc["attention_mask"].to(device)
        if attention_mask.size(1) == 0:
            attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = torch.ones_like(input_ids, device=device)

    cur_ids = input_ids.clone()
    cur_attn = attention_mask.clone()

    for _ in range(max_new_tokens):
        # prevent exceeding training max_length
        if hasattr(model, "max_length") and cur_ids.size(1) >= model.max_length:
            break

        logits = model(cur_ids, attention_mask=cur_attn)  # (1, seq, vocab)
        next_logits = logits[:, -1, :].squeeze(0)

        # repetition penalty
        if repetition_penalty != 1.0:
            for previous_token in set(cur_ids.view(-1).tolist()):
                if next_logits[previous_token] < 0:
                    next_logits[previous_token] *= repetition_penalty
                else:
                    next_logits[previous_token] /= repetition_penalty

        if temperature != 1.0:
            next_logits = next_logits / float(max(1e-8, temperature))

        filtered_logits = top_k_top_p_filtering(next_logits.clone(), top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1,1)
        cur_ids = torch.cat([cur_ids, next_token], dim=1)
        new_attn = torch.ones((cur_attn.size(0), 1), dtype=cur_attn.dtype, device=device)
        cur_attn = torch.cat([cur_attn, new_attn], dim=1)

        if getattr(tokenizer, "eos_token_id", None) is not None and next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = cur_ids[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Generator inference for phishing-email generator")
    parser.add_argument("--ckpt", type=str, default="models/generator_final.pt", help="path to checkpoint")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="path to save generated outputs (txt)")
    args = parser.parse_args()

    gcfg = GeneratorConfig()

    # device selection
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("[LOG] Loading tokenizer...")
    tokenizer = load_tokenizer(gcfg)
    print(f"[LOG] Tokenizer vocab size: {tokenizer.vocab_size}")

    print("[LOG] Building model...")
    model = GeneratorModel(
        vocab_size=tokenizer.vocab_size,
        d_model=gcfg.d_model,
        nhead=gcfg.nhead,
        num_layers=gcfg.num_layers,
        max_length=gcfg.max_length,
        dropout=gcfg.dropout
    )

    # load checkpoint
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"[LOG] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model = safe_load_state(model, state)
    model.to(device)

    # compute prompt token length and cap max_new_tokens to avoid overflow
    enc_prompt = tokenizer(args.prompt, return_tensors="pt", truncation=True, padding=False)
    prompt_len = enc_prompt.get("input_ids", torch.zeros((1,0))).size(1)
    effective_max_new = min(args.max_new_tokens, max(1, model.max_length - prompt_len))
    if effective_max_new < args.max_new_tokens:
        print(f"[WARN] Capping max_new_tokens to {effective_max_new} to avoid exceeding model.max_length ({model.max_length})")

    outputs = []
    print(f"[LOG] Generating {args.n_samples} samples (prompt length={prompt_len})...")
    for i in range(args.n_samples):
        sample = generate_samples(
            model, tokenizer, device,
            prompt=args.prompt,
            max_new_tokens=effective_max_new,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        header = f"\n=== SAMPLE {i+1} ===\n"
        print(header + sample + "\n")
        outputs.append(sample)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for s in outputs:
                f.write(s.strip() + "\n\n")
        print(f"[LOG] Saved {len(outputs)} samples to: {args.out}")

if __name__ == "__main__":
    main()
