import torch
import pandas as pd
import os
import sys
from graph_builder import UrlGraphBuilder
from model import PhishingGNN

# --- CONFIGURATION ---
BATCH_SIZE = 1
AUTOSAVE_INTERVAL = 50 # Lowered this because training is slower now
MODEL_DIR = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
DATASET_PATH = "dataset.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_dataset():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return list(zip(df['url'], df['label']))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n[INFO] STARTING LIVE TRAINING.")
    print("The model will SCAN every single URL in the dataset.")
    print("This will take time. Press Ctrl+C to save and stop.\n")

    builder = UrlGraphBuilder()

    # Model Setup
    model = PhishingGNN(input_dim=4, hidden_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    start_epoch = 0
    processed_count = 0

    # Resume Logic
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Resuming...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        processed_count = checkpoint['processed_count']
        print(f"Resuming from Epoch {start_epoch}, Processed Count {processed_count}")

    raw_data = load_dataset()
    total_samples = len(raw_data)

    model.train()

    try:
        for epoch in range(start_epoch, 5):
            print(f"\n--- Starting Epoch {epoch+1} ---")
            current_batch_idx = 0

            for url, label in raw_data:
                # Skip already processed
                if current_batch_idx < processed_count:
                    current_batch_idx += 1
                    continue

                # --- BUILD GRAPH (ALWAYS ONLINE) ---
                try:
                    # Visual feedback so you know it's not frozen
                    sys.stdout.write(f"\r[Scan {current_batch_idx}] {url[:40]}...")
                    sys.stdout.flush()

                    # No offline flag anymore - it just scans
                    data = builder.get_features(url)

                    data.y = torch.tensor([float(label)], dtype=torch.float)
                    data = data.to(device)

                except Exception as e:
                    # If parsing completely fails, skip
                    current_batch_idx += 1
                    processed_count += 1
                    continue

                # --- TRAIN STEP ---
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out.view(-1), data.y)
                loss.backward()
                optimizer.step()

                current_batch_idx += 1
                processed_count += 1

                # Logging
                if current_batch_idx % 10 == 0: # Log often
                    print(f" Loss: {loss.item():.4f}")

                if current_batch_idx % AUTOSAVE_INTERVAL == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'processed_count': processed_count
                    }, CHECKPOINT_PATH)

            # End of Epoch
            processed_count = 0
            epoch_path = os.path.join(MODEL_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)
            print(f"Epoch {epoch+1} Complete. Model saved.")

    except KeyboardInterrupt:
        print("\n\n!!! KILL SWITCH ACTIVATED (Ctrl+C) !!!")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'processed_count': processed_count
        }, CHECKPOINT_PATH)
        print("Progress saved. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    train()
