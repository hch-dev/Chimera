# train_gan.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs import GeneratorConfig, TrainingConfig
from utils import ensure_dir, log, set_seed
from dataset_loader import load_and_tokenize
from generator_model import GeneratorModel

def collate_batch(batch):
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    attn = torch.stack([b[1] for b in batch], dim=0)
    return input_ids, attn

def main():
    gcfg = GeneratorConfig()
    tcfg = TrainingConfig()
    set_seed(tcfg.seed)

    device = torch.device(gcfg.device if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # 1. Load (New) Dataset
    dataset, tokenizer = load_and_tokenize(gcfg, tcfg)
    dataloader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_batch)

    # 2. Initialize Model Structure
    model = GeneratorModel(max_length=gcfg.max_length)

    # Resize embeddings to match tokenizer (50258)
    new_vocab_size = len(tokenizer)
    model.gpt.resize_token_embeddings(new_vocab_size)

    # --- CRITICAL UPDATE: RESUME LOGIC ---
    if tcfg.resume_checkpoint and os.path.exists(tcfg.resume_checkpoint):
        log(f"♻️  Resuming training from: {tcfg.resume_checkpoint}")
        checkpoint = torch.load(tcfg.resume_checkpoint, map_location="cpu")

        # Handle state dict variations
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        log("✅ Previous 'Brain' loaded successfully. Improving it now...")
    else:
        log("🆕 Starting training from scratch (Base GPT-2)...")
        model.to(device)
    # -------------------------------------

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)

    ensure_dir(tcfg.save_dir)

    log(f"Starting training on {len(dataset)} examples...")
    for epoch in range(1, tcfg.num_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for input_ids, attn in dataloader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]

            logits = model(inputs, attention_mask=attn[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 50 == 0:
                print(f"   Step {steps}/{len(dataloader)} Loss: {loss.item():.4f}", end='\r')

        avg_loss = total_loss / max(1, steps)
        log(f"\n[EPOCH {epoch}/{tcfg.num_epochs}] avg_loss={avg_loss:.4f}")

        # Overwrite the checkpoint so it keeps getting better
        save_path = os.path.join(tcfg.save_dir, "generator_final.pt")
        torch.save(model.state_dict(), save_path)

    log(f"Update complete. Model saved to: {save_path}")

if __name__ == "__main__":
    main()
