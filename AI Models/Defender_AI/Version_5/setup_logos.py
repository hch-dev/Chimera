import os
import requests

LOGO_DIR = "reference_data/logos"
os.makedirs(LOGO_DIR, exist_ok=True)

# PROFESSIONAL TARGET LIST (Top Phishing Vectors 2024-2025)
logos = {
    # --- BIG TECH & CLOUD ---
    "google.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/500px-Google_2015_logo.svg.png",
    "microsoft.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Microsoft_logo_%282012%29.svg/500px-Microsoft_logo_%282012%29.svg.png",
    "apple.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Apple_logo_black.svg/500px-Apple_logo_black.svg.png",
    "amazon.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/600px-Amazon_logo.svg.png",
    "adobe.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Adobe_Corporate_Logo.png/512px-Adobe_Corporate_Logo.png",
    "dropbox.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Dropbox_Icon.svg/512px-Dropbox_Icon.svg.png",

    # --- FINANCIAL & BANKING ---
    "paypal.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/PayPal.svg/500px-PayPal.svg.png",
    "chase.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Chase_Bank_logo.svg/600px-Chase_Bank_logo.svg.png",
    "wells_fargo.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Wells_Fargo_Bank.svg/600px-Wells_Fargo_Bank.svg.png",
    "citi.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Citi.svg/500px-Citi.svg.png",
    "boa.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Bank_of_America_logo_2024.svg/600px-Bank_of_America_logo_2024.svg.png",
    "amex.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/American_Express_logo.svg/600px-American_Express_logo.svg.png",

    # --- SOCIAL MEDIA ---
    "facebook.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b8/2021_Facebook_icon.svg/512px-2021_Facebook_icon.svg.png",
    "instagram.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/500px-Instagram_logo_2016.svg.png",
    "linkedin.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/480px-LinkedIn_logo_initials.png",
    "twitter_x.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/X_icon_2.svg/512px-X_icon_2.svg.png",
    "whatsapp.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/WhatsApp.svg/500px-WhatsApp.svg.png",

    # --- LOGISTICS (Package Theft Scams) ---
    "dhl.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/DHL_Logo.svg/600px-DHL_Logo.svg.png",
    "fedex.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/FedEx_Express.svg/600px-FedEx_Express.svg.png",
    "ups.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/UPS_Logo_Shield_2017.svg/455px-UPS_Logo_Shield_2017.svg.png",

    # --- ENTERTAINMENT & TELECOM ---
    "netflix.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Netflix_2015_logo.svg/600px-Netflix_2015_logo.svg.png",
    "spotify.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/500px-Spotify_logo_without_text.svg.png",
    "att.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/AT%26T_logo_2016.svg/500px-AT%26T_logo_2016.svg.png",
    "verizon.png": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Verizon_2024_logo.svg/640px-Verizon_2024_logo.svg.png"
}

def download_logo(name, url):
    print(f"[*] Fetching resource: {name}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            file_path = os.path.join(LOGO_DIR, name)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"    [✔] Success: {name}")
        else:
            print(f"    [!] Failed {name}: HTTP {response.status_code}")
    except Exception as e:
        print(f"    [X] Critical Error: {e}")

if __name__ == "__main__":
    print(f"=== INITIALIZING DEFENDER V5 ASSET LIBRARY ===")
    print(f"Target Directory: {os.path.abspath(LOGO_DIR)}")
    for name, url in logos.items():
        download_logo(name, url)
    print("\n=== LIBRARY UPDATE COMPLETE ===")