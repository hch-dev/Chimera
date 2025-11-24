"""
Quick dataset validation script for attacker_data.
Checks jsonl structure, field presence and sample counts.
"""

import json

DATASET_PATH = "../attacker_data/train.jsonl"

required_fields = ["input", "target"]

print("=== Checking Dataset ===")
count = 0

try:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            count += 1

            # check fields
            for field in required_fields:
                if field not in obj:
                    print(f"❌ Missing field: {field} in line {count}")

            # show first few samples
            if count <= 3:
                print("\nSample", count)
                print(json.dumps(obj, indent=2))

    print(f"\n✔ Dataset loaded successfully.")
    print(f"✔ Total samples: {count}")

except Exception as e:
    print("❌ Error reading dataset:", e)
