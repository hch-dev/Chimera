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
        self.orb = cv2.ORB_create(nfeatures=2000)

    @staticmethod
    def sanitize_url(url):
        if not url: 
            return ""
        # 1. Strip whitespace
        # 2. Replace backslashes '\' with forward slashes '/' (common copy-paste error)
        url = url.strip().replace("\\", "/")
        
        if not url.startswith(('http://', 'https://', 'file://')):
            return 'https://' + url
        return url

    def check_identity_mismatch(self, screenshot_path, current_url):
        # Basic Validation: Ensure the input looks like a URL (must contain a dot, e.g., 'google.com')
        if not current_url or "." not in current_url:
            log.warning(f"Invalid URL format: '{current_url}'. Skipping visual analysis.")
            return {"mismatch_found": False, "details": "Skipped: Invalid URL provided."}

        # 1. Standardize the URL (Handles 'google.com' input automatically)
        current_url = self.sanitize_url(current_url)

        # Graceful handling: If URL was unreachable, no screenshot exists.
        # Instead of erroring out, we log a warning and skip analysis.
        if not os.path.exists(screenshot_path):
            log.warning(f"Visual Analysis Skipped: Could not load evidence for {current_url}. Site may be unreachable.")
            return {"mismatch_found": False, "details": "Skipped: URL Unreachable or Screenshot Failed."}

        main_image = cv2.imread(screenshot_path)
        if main_image is None: 
            log.warning(f"Visual Analysis Skipped: Image file is corrupted or empty: {screenshot_path}")
            return {"mismatch_found": False, "details": "Skipped: Invalid Screenshot Data."}
            
        gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        
        # --- STRATEGY 1: ORB FEATURE MATCHING (Complex Logos) ---
        orb_match = self._check_orb(gray_main, current_url)
        if orb_match:
            return orb_match

        # --- STRATEGY 2: TEMPLATE MATCHING (Simple/Flat Logos) ---
        # This runs ONLY if ORB failed. It catches Microsoft, Chase, Apple, etc.
        tpl_match = self._check_template(gray_main, current_url)
        if tpl_match:
            return tpl_match

        return {"mismatch_found": False, "details": "Safe: No visual identity mismatch detected."}

    def _check_orb(self, gray_main, current_url):
        try:
            kp_screen, des_screen = self.orb.detectAndCompute(gray_main, None)
            if des_screen is None: return None

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            for logo_file, rule in self.brand_rules.items():
                logo_path = os.path.join("reference_data", logo_file)
                if not os.path.exists(logo_path): continue
                
                template = cv2.imread(logo_path, 0)
                if template is None: continue

                kp_logo, des_logo = self.orb.detectAndCompute(template, None)
                if des_logo is None or len(kp_logo) < 2: continue

                matches = bf.match(des_logo, des_screen)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 10:
                    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_screen[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        inliers = mask.ravel().tolist().count(1)
                        if inliers >= 18:
                            log.info(f"ORB Detection: {rule['name']} (Inliers: {inliers})")
                            return self._verify_domain(rule, current_url)
        except cv2.error as e:
            log.warning(f"OpenCV Error in ORB matching: {e}")
        except Exception as e:
            log.error(f"Unexpected error in ORB matching: {e}")
            
        return None

    def _check_template(self, gray_main, current_url):
        """
        Multi-Scale Template Matching with Strict Filtering.
        """
        for logo_file, rule in self.brand_rules.items():
            logo_path = os.path.join("reference_data", logo_file)
            if not os.path.exists(logo_path): continue
            
            template = cv2.imread(logo_path, 0)
            if template is None: continue

            # Loop through scales: 100%, 80%, 60% ... 20% size of the LOGO
            for scale in np.linspace(0.2, 1.0, 5)[::-1]:
                try:
                    resized_width = int(template.shape[1] * scale)
                    resized_height = int(template.shape[0] * scale)
                    
                    # TUNING 1: Min Size Increased to 50px
                    if resized_width < 50 or resized_height < 50: break

                    resized_tpl = cv2.resize(template, (resized_width, resized_height))

                    # TUNING 4: Template Validity Check
                    if np.std(resized_tpl) < 10:
                        continue

                    # Safety: Template must be smaller than screenshot
                    if resized_tpl.shape[0] > gray_main.shape[0] or resized_tpl.shape[1] > gray_main.shape[1]:
                        continue

                    res = cv2.matchTemplate(gray_main, resized_tpl, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    # TUNING 2: Ultra Strict Threshold (0.96)
                    if max_val > 0.96:
                        # TUNING 3: Variance Check
                        top_left = max_loc
                        h, w = resized_tpl.shape
                        roi = gray_main[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
                        
                        if roi.size > 0:
                            sigma = np.std(roi)
                            if sigma < 10: 
                                continue

                        log.info(f"Template Detection: {rule['name']} (Conf: {max_val:.2f})")
                        return self._verify_domain(rule, current_url)

                except cv2.error as e:
                    # This catches "Assertion failed" or "Invalid Argument" errors
                    # specifically from resizing or matchTemplate
                    log.warning(f"OpenCV Error processing {rule['name']} at scale {scale:.1f}: {e}")
                    continue
                except Exception as e:
                    log.error(f"Unexpected error processing {rule['name']}: {e}")
                    continue

        return None

    def _verify_domain(self, rule, current_url):
        is_safe = any(domain in current_url for domain in rule['safe_domains'])
        if not is_safe:
            return {
                "mismatch_found": True, 
                "details": f"Found {rule['name']} logo on suspicious domain: {current_url}"
            }
        return {
            "mismatch_found": False,
            "details": f"Safe: Found {rule['name']} logo on legitimate domain."
        }