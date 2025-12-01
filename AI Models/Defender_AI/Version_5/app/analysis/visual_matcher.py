import cv2
import json
import os
import numpy as np
from app.utils.logger import setup_logger

log = setup_logger("visual_analyst")

class VisualAnalyst:
    def __init__(self, config_path="reference_data/brands_config.json"):
        with open(config_path, 'r') as f:
            self.brand_rules = json.load(f)
        log.info(f"Loaded visual rules for {len(self.brand_rules)} brands.")

    def check_identity_mismatch(self, screenshot_path, current_url):
        """
        Returns True if a logo is found on a WRONG domain.
        """
        if not os.path.exists(screenshot_path):
            log.error(f"Screenshot not found: {screenshot_path}")
            return {"mismatch_found": False}

        # Load the screenshot
        main_image = cv2.imread(screenshot_path)
        gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

        for logo_file, rule in self.brand_rules.items():
            logo_path = os.path.join("reference_data", logo_file)
            
            # Check if logo exists in our reference folder
            if not os.path.exists(logo_path):
                continue

            # Load Logo
            template = cv2.imread(logo_path, 0)
            
            # MATCHING: Look for the logo inside the screenshot
            result = cv2.matchTemplate(gray_main, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Threshold: 0.8 means 80% confidence match
            if max_val > 0.8:
                log.info(f"Logo Detected: {rule['name']} (Confidence: {max_val:.2f})")
                
                # VERIFICATION: Is the URL safe?
                is_safe = any(domain in current_url for domain in rule['safe_domains'])
                
                if not is_safe:
                    return {
                        "mismatch_found": True, 
                        "details": f"Found {rule['name']} logo on suspicious domain: {current_url}"
                    }
        
        return {"mismatch_found": False}