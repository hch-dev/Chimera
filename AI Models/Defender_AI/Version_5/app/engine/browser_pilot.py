from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os
from app.utils.logger import setup_logger

log = setup_logger("browser_pilot")

class SandboxBrowser:
    def __init__(self, docker_url):
        log.info(f"Connecting to Sandbox Browser at {docker_url}...")
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # Pretend to be a real Windows PC
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        try:
            self.driver = webdriver.Remote(
                command_executor=docker_url,
                options=options
            )
            log.info("Browser connected successfully.")
        except Exception as e:
            log.critical(f"Could not connect to Docker Browser. Is it running? Error: {e}")
            raise e

    def visit_and_capture(self, url):
        """
        Visits the URL, captures evidence, and returns paths.
        """
        try:
            log.info(f"Navigating to {url}...")
            self.driver.get(url)
            
            # Anti-Evasion: Wait & Scroll
            time.sleep(3) 
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)

            # Create storage path
            os.makedirs("storage/screenshots", exist_ok=True)
            screenshot_path = os.path.join("storage", "screenshots", "evidence.png")
            
            # Capture
            self.driver.save_screenshot(screenshot_path)
            final_url = self.driver.current_url
            source = self.driver.page_source
            
            return screenshot_path, final_url, source

        except Exception as e:
            log.error(f"Failed to capture URL: {e}")
            raise e

    def close(self):
        if self.driver:
            self.driver.quit()
            log.info("Browser session closed.")