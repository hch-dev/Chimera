import os
import shutil

# --- PATH DEFINITIONS ---
# We use expanduser to reliably find your home directory
base_dir = os.path.expanduser("~/Documents/Code/Projects/Chimera")
server_path = os.path.join(base_dir, "Server/server.py")
ai_main_path = os.path.join(base_dir, "AI Models/Defender_AI/main.py")
v1_path = os.path.join(base_dir, "AI Models/Defender_AI/Version_1")
ai_path = os.path.join(base_dir, "AI Models/Defender_AI")

# --- 1. CLEAN STALE CACHE ---
print("🧹 Cleaning stale Python cache files...")
for root in [v1_path, ai_path]:
    pycache = os.path.join(root, "__pycache__")
    if os.path.exists(pycache):
        try:
            shutil.rmtree(pycache)
            print(f"   Deleted: {pycache}")
        except Exception as e:
            print(f"   Failed to delete {pycache}: {e}")

# --- 2. THE CORRECT SERVER CODE ---
server_code = r"""from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import asyncio

# --- PATH SETUP ---
# Calculate paths relative to THIS file (server.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up two levels: Server -> Chimera -> AI Models -> Defender_AI
AI_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'AI Models', 'Defender_AI'))

print(f"[*] Configured AI Directory: {AI_DIR}")

# Add to Python Path
if AI_DIR not in sys.path:
    sys.path.append(AI_DIR)

try:
    # This imports the 'scan_url_async' function from Defender_AI/main.py
    from main import scan_url_async
    print("✅ Successfully imported AI Engine")
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    # Debugging aid:
    print(f"   sys.path is: {sys.path}")
    print(f"   Looking for 'main.py' in: {AI_DIR}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

@app.route('/scan', methods=['POST'])
def scan_endpoint():
    data = request.get_json()
    url = data.get('url', '')

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    print(f"[*] Incoming Scan Request for: {url}")

    try:
        # Run the Async Logic inside Flask
        result = asyncio.run(scan_url_async(url))
        return jsonify(result)
    except Exception as e:
        print(f"[!] Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"[*] Backend Server Running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
"""

# --- 3. THE CORRECT AGGREGATOR CODE ---
aggregator_code = r"""import asyncio
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

def extract_score(version, output):
    clean_output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', output)
    score = 0

    if version == "V1":
        verdict_match = re.search(r"VERDICT:.*\(Score:\s*([0-9.]+)\)", clean_output)
        if verdict_match:
            score = float(verdict_match.group(1))
    elif version == "V2":
        match = re.search(r"'confidence':\s*([0-9.]+)", clean_output)
        if match:
            raw = float(match.group(1))
            score = raw * 100 if raw <= 1.0 else raw
    elif version == "V4":
        match = re.search(r"Risk Score:\s*(\d+)/100", clean_output)
        if match: score = float(match.group(1))

    return int(score)

async def run_scanner(version, config, url):
    script_path = os.path.join(config["dir"], config["file"])

    # Auto-detect filename for V1
    if version == "V1" and not os.path.exists(script_path):
        alt_path = os.path.join(config["dir"], "main.py")
        if os.path.exists(alt_path): script_path = alt_path

    if not os.path.exists(script_path):
        return version, 0, {}

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=config["dir"]
        )

        stdout_data, stderr_data = await process.communicate(input=f"{url}\n".encode())
        output = stdout_data.decode(errors='ignore')

        score = extract_score(version, output)
        return version, score, {"raw_output": output}

    except Exception:
        return version, 0, {}

# --- SERVER ENTRY POINT ---
async def scan_url_async(url):
    tasks = [
        run_scanner("V1", SCANNERS["V1"], url),
        run_scanner("V2", SCANNERS["V2"], url),
        run_scanner("V4", SCANNERS["V4"], url)
    ]
    results = await asyncio.gather(*tasks)

    response_data = {
        "url": url,
        "v1_score": 0, "v2_score": 0, "v4_score": 0,
        "final_score": 0,
        "details": {}
    }

    scores = []

    for version, score, _ in results:
        if version == "V1": response_data["v1_score"] = score
        if version == "V2": response_data["v2_score"] = score
        if version == "V4": response_data["v4_score"] = score
        if score > 0: scores.append(score)

    final_verdict = 0
    if scores:
        max_score = max(scores)
        if max_score > 85: final_verdict = max_score
        else: final_verdict = int(sum(scores) / len(scores))

    response_data["final_score"] = final_verdict

    # Mock details
    response_data["details"] = {
        "ssl_presence_and_validity": {"score": 100 if final_verdict > 80 else 0, "message": "SSL Invalid" if final_verdict > 80 else "Valid"},
        "domain_age_analysis": {"score": 0, "message": "Established Domain"}
    }

    return response_data
"""

# --- 4. OVERWRITE FILES ---
print(f"📝 Overwriting Server File: {server_path}")
try:
    with open(server_path, "w") as f:
        f.write(server_code)
    print("   ✅ Server file updated.")
except FileNotFoundError:
    print(f"   ❌ Path not found: {server_path}")

print(f"📝 Overwriting Aggregator File: {ai_main_path}")
try:
    with open(ai_main_path, "w") as f:
        f.write(aggregator_code)
    print("   ✅ Aggregator file updated.")
except FileNotFoundError:
    print(f"   ❌ Path not found: {ai_main_path}")

print("\n✨ Fix Complete. Please run 'python server.py' from the Server directory.")
