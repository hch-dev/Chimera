# train_gan.py
"""
Train the generator (lightweight LM) on RonakAJ/phising_email.
This trains a causal LM (not a true GAN) — practical and stable for building a generator
that produces grammatical phishing-like emails for defender training.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs import GeneratorConfig, TrainingConfig
from utils import ensure_dir, log, set_seed
from dataset_loader import load_and_tokenize
from generator_model import GeneratorModel

def collate_batch(batch):
    # batch is list of (input_ids, attention_mask)
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attn = torch.stack([b[1] for b in batch], dim=0)
    return input_ids, attn

def main():
    gcfg = GeneratorConfig()
    tcfg = TrainingConfig()
    set_seed(tcfg.seed)

    device = torch.device(gcfg.device if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    dataset, tokenizer = load_and_tokenize(gcfg, tcfg)
    dataloader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_batch)

    vocab_size = tokenizer.vocab_size
    log(f"Tokenizer vocab size: {vocab_size}")

    model = GeneratorModel(vocab_size=vocab_size,
                           d_model=gcfg.d_model,
                           nhead=gcfg.nhead,
                           num_layers=gcfg.num_layers,
                           max_length=gcfg.max_length,
                           dropout=gcfg.dropout).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)

    ensure_dir(tcfg.save_dir)

    log("Starting training...")
    for epoch in range(1, tcfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for input_ids, attn in dataloader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            # shift inputs for causal loss: predict next token
            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            logits = model(inputs, attention_mask=attn[:, :-1])
            # logits: (batch, seq-1, vocab)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(1, steps)
        log(f"[EPOCH {epoch}/{tcfg.num_epochs}] avg_loss={avg_loss:.4f}")

        # save checkpoint each epoch
        save_path = os.path.join(tcfg.save_dir, f"generator_epoch{epoch}.pt")
        torch.save({"model_state_dict": model.state_dict(),
                    "tokenizer_name": gcfg.model_name}, save_path)
        log(f"Saved model checkpoint: {save_path}")

    # final save
    final_path = os.path.join(tcfg.save_dir, "generator_final.pt")
    torch.save({"model_state_dict": model.state_dict(),
                "tokenizer_name": gcfg.model_name}, final_path)
    log(f"Training complete. Final model saved to: {final_path}")

if __name__ == "__main__":
    main()
