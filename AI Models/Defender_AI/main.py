import asyncio
import os
import sys
import re

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(BASE_DIR, "Version_1")
V2_DIR = os.path.join(BASE_DIR, "Version_2")
V4_DIR = os.path.join(BASE_DIR, "Version_4")

SCANNERS = {
    "V1": {"dir": V1_DIR, "file": "main_async.py"},
    "V2": {"dir": V2_DIR, "file": "main.py"},
    "V4": {"dir": V4_DIR, "file": "main.py"},
}

# --- IMPROVED PARSING LOGIC ---
def extract_score(version, output):
    """
     robustly extracts the FINAL score, ignoring intermediate logs.
    """
    # 1. Clean ANSI colors
    clean_output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)

    score = 0

    if version == "V1":
        # PRIORITY 1: Find the Final Verdict line (e.g., "Score: 90.25")
        # Matches: "VERDICT: PHISHING DETECTED (Score: 90.25)"
        verdict_match = re.search(r"VERDICT:.*\(Score:\s*([0-9.]+)\)", clean_output)

        if verdict_match:
            score = float(verdict_match.group(1))
        else:
            # Fallback: Look for "Risk X/100" but verify it's the Total, not a feature.
            # (V1 doesn't usually print "Risk X/100" for the total, so we rely on VERDICT)
            pass

    elif version == "V2":
        # Matches dictionary format: 'confidence': 0.85
        match = re.search(r"'confidence':\s*([0-9.]+)", clean_output)
        if match:
            raw_score = float(match.group(1))
            score = raw_score * 100 if raw_score <= 1.0 else raw_score

    elif version == "V4":
        # Matches "Risk Score: 68/100"
        match = re.search(r"Risk Score:\s*(\d+)/100", clean_output)
        if match: score = float(match.group(1))

    return int(score)

# --- ASYNC PROCESS RUNNER ---
async def run_scanner(version, config, url):
    script_path = os.path.join(config["dir"], config["file"])

    # Auto-detect filename for V1 if "main_async.py" doesn't exist
    if version == "V1" and not os.path.exists(script_path):
        alt_path = os.path.join(config["dir"], "main.py")
        if os.path.exists(alt_path):
            script_path = alt_path

    if not os.path.exists(script_path):
        return version, None, f"File not found: {script_path}"

    print(f"   [+] Launching {version}...")

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=config["dir"]
        )

        stdout_data, stderr_data = await process.communicate(input=f"{url}\n".encode())

        stdout_str = stdout_data.decode(errors='ignore')
        stderr_str = stderr_data.decode(errors='ignore')

        if process.returncode != 0:
            return version, None, f"Crash (Code {process.returncode}): {stderr_str.strip()[-100:]}"

        score = extract_score(version, stdout_str)
        return version, score, stdout_str

    except Exception as e:
        return version, None, f"Execution Error: {str(e)}"

# --- MAIN AGGREGATOR ---
async def main_async():
    print("\n" + "="*60)
    print(f"🚀 CHIMERA DEFENSE PROTOCOL: PARALLEL SCAN")
    print("="*60)

    try:
        target_url = input("🎯 Enter Target URL: ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    print("-" * 60)

    # 1. Run Scans
    tasks = [
        run_scanner("V1", SCANNERS["V1"], target_url),
        run_scanner("V2", SCANNERS["V2"], target_url),
        run_scanner("V4", SCANNERS["V4"], target_url)
    ]
    results = await asyncio.gather(*tasks)

    # 2. Report
    print("\n" + "="*60)
    print("📊 INTEGRATED REPORT")
    print("="*60)

    scores = []

    for version, score, debug_log in results:
        if score is None:
            print(f"❌ {version}: FAILED")
            print(f"   └── Reason: {debug_log}")
        else:
            status = "🔴" if score > 75 else "⚠️ " if score > 50 else "✅"
            print(f"{status} {version}: {score}/100")
            scores.append(score)

    # 3. Final Verdict
    final_verdict = 0
    if scores:
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Smart weighting: If any model is highly confident (>85), trust it.
        # Otherwise, average them out.
        if max_score > 85:
            final_verdict = max_score
            print(f"\n   [!] Critical Alert: One engine detected high risk.")
        else:
            final_verdict = int(avg_score)

    print("-" * 60)
    risk_label = "✅ SAFE"
    if final_verdict > 75: risk_label = "❌ BLOCKED"
    elif final_verdict > 50: risk_label = "⚠️ SUSPICIOUS"

    print(f"🛡️  FINAL VERDICT: {risk_label} ({final_verdict}/100)")
    print("="*60 + "\n")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
