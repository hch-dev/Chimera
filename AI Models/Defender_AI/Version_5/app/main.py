print("APP INITIATED")
import sys
import json
import os
from app.engine.browser_pilot import SandboxBrowser
from app.analysis.visual_matcher import VisualAnalyst
from app.utils.logger import setup_logger

# Initialize Logger
log = setup_logger("main_controller")

def run_investigation(target_url):
    log.info(f"[*] Starting V5 Sandbox Investigation for: {target_url}")
    
    pilot = None
    try:
        # 1. Initialize Components
        # Connect to Docker (ensure docker-compose up is running)
        pilot = SandboxBrowser(docker_url="http://localhost:4444/wd/hub")
        analyst = VisualAnalyst(config_path="reference_data/brands_config.json")
        
        results = {
            "url": target_url,
            "phishing_detected": False,
            "evidence": []
        }

        # 2. Detonate URL
        log.info("Detonating URL in isolated container...")
        print("   [~] Opening browser in sandbox...")
        screenshot_path, final_url, dom_source = pilot.visit_and_capture(target_url)
        
        results['final_url'] = final_url
        log.info(f"Landed on: {final_url}")
        
        # 3. Analyze Evidence (Visual Match)
        log.info("Analyzing visual evidence...")
        visual_verdict = analyst.check_identity_mismatch(screenshot_path, final_url)
        
        print("-" * 50)
        if visual_verdict['mismatch_found']:
            results['phishing_detected'] = True
            results['evidence'].append(visual_verdict['details'])
            log.warning(f"PHISHING DETECTED: {visual_verdict['details']}")
            print(f" [!] DANGER: Phishing Detected!")
            print(f" [!] Reason: {visual_verdict['details']}")
        else:
            log.info("Visual check passed. No mismatch found.")
            print(f" [+] CLEAN: No visual spoofing detected for {target_url}")
        print("-" * 50)

        # Save Report
        os.makedirs("storage/reports", exist_ok=True)
        report_path = f"storage/reports/scan_{int(os.path.getmtime(screenshot_path))}.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=4)
            
        return results

    except Exception as e:
        log.error(f"Error during sandbox execution: {e}")
        print(f" [X] Error: {e}")
    
    finally:
        # 4. Cleanup
        if pilot:
            pilot.close()

if __name__ == "__main__":
    print("=========================================")
    print("   DEFENDER V5: SANDBOX ENGINE ACTIVE    ")
    print("=========================================")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter URL to scan (or type 'exit'): ").strip()
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("[*] Shutting down Sandbox Engine. Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
                
            # Run the scan
            run_investigation(user_input)
            
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n[*] Force quit detected. Exiting...")
            break