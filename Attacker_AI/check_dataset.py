import json

def check_file(path):
    print(f"\nChecking {path}...")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
            except:
                print(f" ❌ Invalid JSON at line {i}")
                return
        print(" ✅ Valid JSONL structure")

check_file("attacker_data/train.jsonl")
check_file("attacker_data/val.jsonl")
