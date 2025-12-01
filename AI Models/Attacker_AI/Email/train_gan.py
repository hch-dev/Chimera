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

    # 1. Load Data
    dataset, tokenizer = load_and_tokenize(gcfg, tcfg)
    dataloader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_batch)

    # 2. Init Model
    model = GeneratorModel(max_length=gcfg.max_length)
    new_vocab_size = len(tokenizer)
    model.gpt.resize_token_embeddings(new_vocab_size)

    # Setup Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    ensure_dir(tcfg.save_dir)

    # --- 3. SMART RESUME LOGIC ---
    start_epoch = 1
    steps_to_skip = 0

    # File paths
    interrupted_path = os.path.join(tcfg.save_dir, "generator_interrupted.pt")
    final_path = os.path.join(tcfg.save_dir, "generator_final.pt")

    # Priority 1: Check for an interrupted session (Force Stopped)
    if os.path.exists(interrupted_path):
        log(f"⚠️  Found interrupted session! Resuming from: {interrupted_path}")
        ckpt = torch.load(interrupted_path, map_location="cpu")

        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        steps_to_skip = ckpt['steps_finished']

        model.to(device)
        log(f"⏩ Fast-forwarding to Epoch {start_epoch}, Batch {steps_to_skip}...")

    # Priority 2: Check for a finished model (Incremental Learning)
    elif tcfg.resume_checkpoint and os.path.exists(tcfg.resume_checkpoint):
        log(f"♻️  Resuming from previous best model: {tcfg.resume_checkpoint}")
        checkpoint = torch.load(tcfg.resume_checkpoint, map_location="cpu")

        # Handle different save formats
        state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state)
        model.to(device)
        log("✅ Previous knowledge loaded! Improving current model.")

    else:
        log("🆕 Starting training from scratch...")
        model.to(device)

    # --- CONFIG FOR AUTO-SAVE ---
    # Calculate how many steps equal 2000 emails
    # Example: 2000 emails / Batch Size 4 = 500 Steps
    save_interval_steps = 2000 // tcfg.batch_size
    if save_interval_steps < 1: save_interval_steps = 1

    log(f"Auto-save enabled: Saving every {save_interval_steps} batches ({save_interval_steps * tcfg.batch_size} emails).")

    model.train()

    # 4. ROBUST TRAINING LOOP
    try:
        for epoch in range(start_epoch, tcfg.num_epochs + 1):
            total_loss = 0.0
            steps = 0

            # If we started a NEW epoch, reset the skip counter
            if epoch > start_epoch:
                steps_to_skip = 0

            for i, (input_ids, attn) in enumerate(dataloader):
                # --- SKIP LOGIC (Fast Forward) ---
                if i < steps_to_skip:
                    if i % 100 == 0: print(f"⏩ Skipping batch {i}/{len(dataloader)}...", end='\r')
                    continue
                # ---------------------------------

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

                # Progress Log
                if i % 50 == 0:
                    print(f"   Epoch {epoch} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}", end='\r')

                # --- AUTO-SAVE (Every 2000 Emails) ---
                if (i + 1) % save_interval_steps == 0:
                    # We save to the 'interrupted' file so if it crashes now, we resume from here
                    state = {
                        'epoch': epoch,
                        'steps_finished': i + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }
                    torch.save(state, interrupted_path)
                    # Print a new line so we don't overwrite the progress bar
                    print(f"\n💾 [AUTO-SAVE] Progress saved at {i} batches ({i * tcfg.batch_size} emails).")
                # -------------------------------------

            avg_loss = total_loss / max(1, steps)
            log(f"\n[EPOCH {epoch}/{tcfg.num_epochs}] avg_loss={avg_loss:.4f}")

            # Save regular checkpoint at end of epoch
            torch.save(model.state_dict(), final_path)

            # Since we finished the epoch safely, we can delete the resume file
            if os.path.exists(interrupted_path):
                os.remove(interrupted_path)

        log(f"Training complete. Final model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\n\n🛑 FORCE STOP DETECTED! Saving state...")

        state = {
            'epoch': epoch,
            'steps_finished': i, # Save exactly where we stopped
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(state, interrupted_path)
        print(f"✅ Progress saved to: {interrupted_path}")
        print(f"   Next time you run this script, it will auto-resume from Epoch {epoch}, Batch {i}.")
        sys.exit(0)

if __name__ == "__main__":
    main()
