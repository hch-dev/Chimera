import torch
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
from urllib.parse import urlparse
# --- 1. CONFIGURATION ---
# Get the directory where THIS file (predict.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the models folder
MODEL_PATH = os.path.join(BASE_DIR, "models", "v3_phishing_roberta")
LOG_FILE = os.path.join(BASE_DIR, "logs", "inference.log")

# Trusted Domains that AI should NEVER flag
WHITELIST = [
    "google.com", "www.google.com", "accounts.google.com", "drive.google.com",
    "microsoft.com", "www.microsoft.com", "login.microsoftonline.com",
    "apple.com", "www.apple.com", "icloud.com",
    "yahoo.com", "www.yahoo.com", "login.yahoo.com",
    "amazon.com", "www.amazon.com",
    "ebay.com", "www.ebay.com",
    "walmart.com", "www.walmart.com",
    "netflix.com", "www.netflix.com",
    "facebook.com", "www.facebook.com",
    "instagram.com", "www.instagram.com",
    "twitter.com", "x.com",
    "linkedin.com", "www.linkedin.com",
    "paypal.com", "www.paypal.com",
    "chase.com", "www.chase.com",
    "bankofamerica.com", "www.bankofamerica.com",
    "wellsfargo.com", "www.wellsfargo.com",
    "americanexpress.com", "www.americanexpress.com"
]

# --- 2. LOGGING SETUP ---
if not os.path.exists('./logs'):
    os.makedirs('./logs')

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler(LOG_FILE, mode='a', maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- 3. LOAD ENGINE ---
logging.getLogger("transformers").setLevel(logging.ERROR)

logger.info("Loading Version 3 Engine...")
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    if os.path.exists(MODEL_PATH):
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        model.to(device)
        logger.info("Engine loaded successfully.")
    else:
        logger.error(f"Model not found at {MODEL_PATH}")
        print("\n❌ Error: Model files missing. Train the model first.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    print(f"\n❌ Error loading model: {e}")

# --- 4. PREPROCESSING & LOGIC ---
def get_domain(url):
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except:
        return ""

def preprocess_input(text):
    return text.replace('/', ' ').replace('-', ' ').replace('.', ' ').replace('_', ' ')

def analyze_text(text):
    if model is None:
        return 0.0, "ERROR: Model not loaded"

    start_time = time.time()
    domain = get_domain(text)
    
    # CHECK 1: Whitelist
    if domain in WHITELIST:
        logger.info(f"Input: '{text}' | Whitelisted Domain ✅ | Verdict: SAFE")
        return 0.0, "SAFE (Whitelisted) ✅"

    # CHECK 2: AI Inference
    clean_text = preprocess_input(text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        
    phishing_score = probs[0][1].item()
    latency = time.time() - start_time
    snippet = text[:50].replace('\n', ' ') + "..."

    # Thresholds
    if phishing_score > 0.75:
        verdict = "HIGH RISK 🚨"
        log_level = "WARNING"
    elif phishing_score > 0.35:
        verdict = "SUSPICIOUS ⚠️"
        log_level = "INFO"
    else:
        verdict = "SAFE ✅"
        log_level = "INFO"
        
    log_msg = f"Input: '{snippet}' | Score: {phishing_score:.4f} | Verdict: {verdict} | Latency: {latency:.3f}s"
    
    if log_level == "WARNING":
        logger.warning(log_msg)
    else:
        logger.info(log_msg)
        
    return phishing_score, verdict

# --- 5. THE RUN FUNCTION (Single Scan Mode) ---
def run():
    print("\n" + "="*40)
    print("          DEFENDER V3: Athena    ")
    print("="*40)

    if model is None:
        print("⚠️  System operating in DIAGNOSTIC MODE (No Model)")
        input("\nPress Enter to return to menu...")
        return
    
    try:
        # 1. Ask for Input ONCE
        user_input = input("\nEnter URL or Text to scan: ").strip()
        
        # 2. Check for exit or empty
        if not user_input or user_input.lower() in ['exit', 'quit', 'q']:
            print("Scan cancelled.")
            return # Return to Main Menu

        # 3. Analyze
        print("Analyzing...")
        score, verdict = analyze_text(user_input)
        
        # 4. Print Result
        print("-" * 40)
        print(f"Input:  {user_input[:60]}...")
        print(f"Result: {verdict} (Risk: {score:.1%})")
        print("-" * 40)

    except KeyboardInterrupt:
        print("\nForce Quit.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # 5. Pause so user can read result
    input("\nPress Enter to return to Main Menu...")

if __name__ == "__main__":
    run()
