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

# --- PARSING LOGIC ---
def extract_score(version, output):
    clean_output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)
    score = 0
    if version == "V1":
        verdict_match = re.search(r"VERDICT:.*\(Score:\s*([0-9.]+)\)", clean_output)
        if verdict_match: score = float(verdict_match.group(1))
    elif version == "V2":
        match = re.search(r"'confidence':\s*([0-9.]+)", clean_output)
        if match:
            raw_score = float(match.group(1))
            score = raw_score * 100 if raw_score <= 1.0 else raw_score
    elif version == "V4":
        match = re.search(r"Risk Score:\s*(\d+)/100", clean_output)
        if match: score = float(match.group(1))
    return int(score)

# --- ASYNC RUNNER ---
async def run_scanner(version, config, url):
    script_path = os.path.join(config["dir"], config["file"])
    if version == "V1" and not os.path.exists(script_path):
        alt_path = os.path.join(config["dir"], "main.py")
        if os.path.exists(alt_path): script_path = alt_path

    if not os.path.exists(script_path):
        return version, None, f"File not found: {script_path}"

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

        score = extract_score(version, stdout_str)
        return version, score, stdout_str
    except Exception as e:
        return version, None, str(e)

# --- MAIN AGGREGATOR FUNCTION ---
async def analyze_url_async(target_url):
    tasks = [
        run_scanner("V1", SCANNERS["V1"], target_url),
        run_scanner("V2", SCANNERS["V2"], target_url),
        run_scanner("V4", SCANNERS["V4"], target_url)
    ]
    results = await asyncio.gather(*tasks)

    scores = []
    details_log = {}

    for version, score, debug_log in results:
        details_log[version] = debug_log
        if score is not None:
            scores.append(score)

    final_verdict = 0
    if scores:
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        final_verdict = max_score if max_score > 85 else int(avg_score)

    return {
        "final_score": final_verdict,
        "confidence": final_verdict,
        "details": {
            "ssl_presence_and_validity": {"message": f"Scanned by V1, V2, V4", "score": 0},
            "threat_intelligence": {"message": f"Engines detected risk: {final_verdict}%", "score": final_verdict},
        }
    }

# --- CRITICAL FIX: ASYNC BRIDGE ---
async def analyze_data(url):
    """
    CHANGED: This is now an 'async' function.
    It awaits the result directly instead of using asyncio.run()
    """
    return await analyze_url_async(url)

if __name__ == "__main__":
    # For local testing only
    print(asyncio.run(analyze_data("http://google.com")))
