import pandas as pd
import os
import sys
import hashlib  # <--- NEW: For stable filenames
from src.utils import capture_screenshot, logger

# CONFIG
PHISHING_CSV = "dataset/raw_urls/phishing.csv" 
LEGIT_CSV = "dataset/raw_urls/legit.csv"
DATASET_DIR = "dataset/train"
MAX_IMAGES_PER_CLASS = 2000  # <--- LIMIT SET HERE

def get_stable_filename(url):
    """Generates a consistent filename based on the URL."""
    # This creates a unique fingerprint (MD5) that never changes
    return hashlib.md5(url.encode('utf-8')).hexdigest() + ".png"

def build_class_data(csv_path, label_name):
    save_dir = os.path.join(DATASET_DIR, label_name)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        logger.error(f"Missing CSV: {csv_path}")
        return

    # Load and Limit URLs immediately
    try:
        
        urls = pd.read_csv(csv_path)['url'].tolist()[:MAX_IMAGES_PER_CLASS]
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {e}")
        return

    logger.info(f"--- Processing {label_name} (Limit: {len(urls)}) ---")
    
    count = 0
    skipped = 0
    
    for url in urls:
        try:
            url = str(url).strip() # Remove spaces
            if not url.startswith('http'):
                url = "https://" + url
            # 1. Generate STABLE filename
            filename = get_stable_filename(url)
            full_path = os.path.join(save_dir, filename)
            
            # 2. Check existence
            if os.path.exists(full_path):
                # Print less frequently to keep terminal clean
                # print(f"Skipping: {url}") 
                skipped += 1
                continue
            
            # 3. Capture
            if capture_screenshot(url, full_path):
                count += 1
                if count % 5 == 0: 
                    print(f"[{label_name}] Captured: {count} | Skipped: {skipped}")
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping safely...")
            sys.exit()
        except Exception as e:
            logger.error(f"Error on {url}: {e}")
    
    logger.info(f"Finished {label_name}. Total New: {count}. Total Skipped: {skipped}.")

if __name__ == "__main__":
    # It will run Phishing first, then Legit
    build_class_data(PHISHING_CSV, "phishing")
    build_class_data(LEGIT_CSV, "legit")