# train_gan.py
import os
import sys
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

    # 1. Load Dataset
    dataset, tokenizer = load_and_tokenize(gcfg, tcfg)
    dataloader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_batch)

    # 2. Init Model
    model = GeneratorModel(max_length=gcfg.max_length)
    new_vocab_size = len(tokenizer)
    model.gpt.resize_token_embeddings(new_vocab_size)

    # Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    ensure_dir(tcfg.save_dir)

    # --- SMART RESUME LOGIC ---
    start_epoch = 1
    steps_to_skip = 0

    interrupted_ckpt = os.path.join(tcfg.save_dir, "generator_interrupted.pt")

    if os.path.exists(interrupted_ckpt):
        log(f"⚠️  Found interrupted session! Resuming from: {interrupted_ckpt}")
        ckpt = torch.load(interrupted_ckpt, map_location="cpu")
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        steps_to_skip = ckpt['steps_finished']
        model.to(device)
        log(f"⏩ Fast-forwarding: Skipping first {steps_to_skip} batches...")

    elif tcfg.resume_checkpoint and os.path.exists(tcfg.resume_checkpoint):
        log(f"♻️  Resuming training from: {tcfg.resume_checkpoint}")
        ckpt = torch.load(tcfg.resume_checkpoint, map_location="cpu")
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state)
        model.to(device)
        log("✅ Previous knowledge loaded.")
    else:
        log("🆕 Starting training from scratch (Fine-Tuning)...")
        model.to(device)

    # Auto-Save every 2000 emails (500 batches)
    save_interval_steps = 2000 // tcfg.batch_size
    if save_interval_steps < 1: save_interval_steps = 1

    model.train()

    try:
        for epoch in range(start_epoch, tcfg.num_epochs + 1):
            total_loss = 0.0
            steps = 0

            if epoch > start_epoch:
                steps_to_skip = 0

            for i, (input_ids, attn) in enumerate(dataloader):
                if i < steps_to_skip:
                    if i % 500 == 0: print(f"⏩ Skipping batch {i}...", end='\r')
                    continue

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

                if i % 50 == 0:
                    print(f"   Epoch {epoch} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}", end='\r')

                if (i + 1) % save_interval_steps == 0:
                    state = {
                        'epoch': epoch,
                        'steps_finished': i + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(state, interrupted_ckpt)
                    print(f"\n💾 [AUTO-SAVE] Progress saved at batch {i}.")

            avg_loss = total_loss / max(1, steps)
            log(f"\n[EPOCH {epoch}/{tcfg.num_epochs}] avg_loss={avg_loss:.4f}")

            save_path = os.path.join(tcfg.save_dir, "generator_final.pt")
            torch.save(model.state_dict(), save_path)

            if os.path.exists(interrupted_ckpt):
                os.remove(interrupted_ckpt)

        log(f"Training complete. Final model saved to: {save_path}")

    except KeyboardInterrupt:
        print("\n\n🛑 FORCE STOP DETECTED! Saving state...")
        state = {
            'epoch': epoch,
            'steps_finished': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(state, interrupted_ckpt)
        print(f"✅ Progress saved to: {interrupted_ckpt}")
        sys.exit(0)

if __name__ == "__main__":
    main()
