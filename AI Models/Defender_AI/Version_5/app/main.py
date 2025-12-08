print("APP INITIATED")
import sys
import json
import os
# --- Imports from your app ---
from app.engine.browser_pilot import SandboxBrowser
from app.analysis.visual_matcher import VisualAnalyst
from app.utils.logger import setup_logger

log = setup_logger("main_controller")

def run_investigation(target_url):
    log.info(f"[*] Starting V5 Sandbox Investigation for: {target_url}")
    pilot = None
    try:
        pilot = SandboxBrowser(docker_url="http://localhost:4444/wd/hub")
        analyst = VisualAnalyst(config_path="reference_data/brands_config.json")
        
        results = {"url": target_url, "phishing_detected": False, "evidence": []}

        print("   [~] Opening browser in sandbox...")
        screenshot_path, final_url, dom_source = pilot.visit_and_capture(target_url)
        
        visual_verdict = analyst.check_identity_mismatch(screenshot_path, final_url)
        
        if visual_verdict['mismatch_found']:
            results['phishing_detected'] = True
            print(f" [!] DANGER: Phishing Detected!")
        else:
            print(f" [+] CLEAN: No visual spoofing detected.")

        return results
    except Exception as e:
        log.error(f"Error: {e}")
        print(f" [X] Error: {e}")
    finally:
        if pilot: pilot.close()

# --- THE RUN FUNCTION (Single Scan Mode) ---
def run():
    print("=========================================")
    print(" DEFENDER V5: Tartarus Sandbox Activated ")
    print("=========================================")
    
    try:
        # 1. Ask for input ONCE
        user_input = input("\nEnter URL to scan: ").strip()
        
        # 2. Check for exit or empty input
        if not user_input or user_input.lower() in ['exit', 'quit', 'q']:
            print("Scan cancelled.")
            return # Returns to Main Menu immediately

        # 3. Run the investigation logic
        run_investigation(user_input)

    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    
    # 4. Pause so the user can read the results
    input("\nPress Enter to return to Main Menu...")

if __name__ == "__main__":
    run()