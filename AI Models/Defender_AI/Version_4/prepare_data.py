import pandas as pd

# --- CONFIGURATION ---
# Replace this with the filename you actually downloaded
DOWNLOADED_FILE = "dataset.csv"

def clean_data():
    print(f"Reading {DOWNLOADED_FILE}...")

    # 1. Load the raw file (Handling bad lines)
    try:
        df = pd.read_csv(DOWNLOADED_FILE, on_bad_lines='skip')
    except:
        # Fallback for datasets that might use different encoding
        df = pd.read_csv(DOWNLOADED_FILE, encoding='latin-1', on_bad_lines='skip')

    print(f"Original Columns: {df.columns.tolist()}")

    # 2. Rename columns to 'url' and 'label'
    # YOU MUST EDIT THIS PART to match your downloaded file's column names
    # Example: If your file has 'URL' and 'Label', change 'URL' to 'url', etc.

    # Standardizing common column names
    df.columns = [c.lower() for c in df.columns] # make everything lowercase

    # Rename common variations to our standard
    if 'status' in df.columns:
        df = df.rename(columns={'status': 'label'})
    if 'phishing' in df.columns: # Sometimes label is called 'phishing'
        df = df.rename(columns={'phishing': 'label'})

    # 3. Standardize Labels to 0 and 1
    # Many datasets use "good"/"bad" or "legitimate"/"phishing" strings.
    # We need integers: 0 = Safe, 1 = Phish

    unique_labels = df['label'].unique()
    print(f"Found labels: {unique_labels}")

    # Map common string labels to 0/1
    label_mapping = {
        'good': 0, 'benign': 0, 'legitimate': 0, 'safe': 0, 0: 0,
        'bad': 1, 'phishing': 1, 'malicious': 1, 'malware': 1, 1: 1
    }

    df['label'] = df['label'].map(label_mapping)

    # Drop rows where label conversion failed
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # 4. Final Cleanup
    final_df = df[['url', 'label']]

    # 5. Save as dataset.csv
    final_df.to_csv("dataset.csv", index=False)
    print(f"Success! Saved {len(final_df)} rows to 'dataset.csv'.")
    print("You can now run 'python train.py'")

if __name__ == "__main__":
    clean_data()
