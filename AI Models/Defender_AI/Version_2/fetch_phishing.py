import pandas as pd
import requests
import io
import os

# --- CONFIGURATION ---
OUTPUT_FILE = "dataset/raw_urls/phishing.csv"
LIMIT = 2000
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_openphish():
    print("⬇️  Fetching OpenPhish...")
    try:
        url = "https://openphish.com/feed.txt"
        response = requests.get(url, headers=HEADERS, timeout=10)
        urls = response.text.strip().split('\n')
        return pd.DataFrame(urls, columns=['url'])
    except:
        return pd.DataFrame()

def get_phishtank():
    print("⬇️  Fetching PhishTank...")
    try:
        url = "http://data.phishtank.com/data/online-valid.csv"
        s = requests.get(url, headers=HEADERS, timeout=15).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        return df[['url']]
    except:
        return pd.DataFrame()

def get_phishstats():
    print("⬇️  Fetching PhishStats...")
    try:
        url = "https://phishstats.info/phish_score.csv"
        df = pd.read_csv(url, skiprows=9, names=['Date','Score','URL','IP'], on_bad_lines='skip')
        return df[['URL']].rename(columns={'URL': 'url'})
    except:
        return pd.DataFrame()

def main():
    print(f"🚀 STARTING DOWNLOAD (Target: {LIMIT} URLs)...")
    
    # 1. Gather Data
    df1 = get_openphish()
    df2 = get_phishtank()
    df3 = get_phishstats()
    
    # 2. Merge & Clean
    print("🔄 Combining and Cleaning...")
    combined = pd.concat([df1, df2, df3], ignore_index=True)
    combined.dropna(inplace=True)
    combined.drop_duplicates(subset=['url'], inplace=True)
    
    # 3. Cut to Limit
    final_df = combined.head(LIMIT)
    
    # 4. Save
    # Ensure folder exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Save to CSV with 'url' header
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print(f"✅ SUCCESS! Saved to: {OUTPUT_FILE}")
    print(f"📄 Header: 'url'")
    print(f"🔢 Total Rows: {len(final_df)}")
    print("="*40)

if __name__ == "__main__":
    main()