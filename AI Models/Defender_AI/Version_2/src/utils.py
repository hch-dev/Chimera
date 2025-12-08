import logging
import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 1. Logger Setup
def setup_logger(name="PHISHING_V2"):
    if not os.path.exists('logs'): os.makedirs('logs')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): return logger
    
    file_handler = logging.FileHandler('logs/app.log')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# 2. Stealth Screenshot Tool
def capture_screenshot(url, save_path):
    options = webdriver.ChromeOptions()
    
    # --- STEALTH MODE SETTINGS ---
    # Use the new headless mode (Indistinguishable from real Chrome)
    options.add_argument('--headless=new') 
    
    # Set a real window size (Servers check this!)
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')
    
    # Hide "AutomationControlled" flag (Crucial for Google/Facebook)
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Spoof User-Agent (Look like a regular Windows laptop)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36")
    # -----------------------------

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        # Prevent detection via navigator.webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        driver.set_page_load_timeout(45) # Give legit sites 15s (they are heavier)
        driver.get(url)
        
        # Wait a bit for animations/popups to settle
        time.sleep(2) 
        
        driver.save_screenshot(save_path)
        driver.quit()
        return True
        
    except Exception as e:
        # Show specific error in console so we know WHY it failed
        logger.error(f"Failed {url}: {str(e)[:100]}...") # Print first 100 chars of error
        if driver:
            try: driver.quit()
            except: pass
        return False