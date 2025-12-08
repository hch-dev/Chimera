# prepare_csvs.py (Placed in Root)
import pandas as pd
import requests
import os

# Ensure the folder exists
os.makedirs("dataset/raw_urls", exist_ok=True)

# --- PART 1: PHISHING (Let Python do this for you) ---
print("Downloading Phishing Data...")
try:
    url = "https://openphish.com/feed.txt"
    response = requests.get(url)
    data = response.text.strip().split('\n')
    
    # Save as CSV with header 'url'
    df = pd.DataFrame(data, columns=['url'])
    df.to_csv("dataset/raw_urls/phishing.csv", index=False)
    print(f"✅ Success! Saved {len(df)} phishing URLs.")
except Exception as e:
    print(f"❌ Error getting phishing data: {e}")

# --- PART 2: LEGIT (Manual Check) ---
if os.path.exists("dataset/raw_urls/legit.csv"):
    print("✅ Found your manually created legit.csv!")
else:
    print("⚠️ WARNING: legit.csv is missing. Please save your Excel file to 'dataset/raw_urls/legit.csv'")