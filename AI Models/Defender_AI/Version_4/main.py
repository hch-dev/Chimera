import sys
import os

# --- 1. SAFE IMPORTS ---
try:
    import torch
    try:
        from graph_builder import UrlGraphBuilder
        from model import PhishingGNN
    except ImportError:
        # Silently fail or minimal error if running automated
        class UrlGraphBuilder: pass
        class PhishingGNN: pass

except ImportError:
    print("⚠️  Warning: PyTorch not found. GNN Engine disabled.")
    torch = None

MODEL_PATH = "models/latest_checkpoint.pth"

# --- 2. LOAD ENGINE ---
def load_brain():
    if torch is None: return None, None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check absolute path relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_model_path = os.path.join(base_dir, MODEL_PATH)

    if not os.path.exists(full_model_path):
        # Try local path just in case
        if os.path.exists(MODEL_PATH):
            full_model_path = MODEL_PATH
        else:
            print(f"❌ Error: Model file not found at {full_model_path}")
            return None, None

    try:
        model = PhishingGNN(input_dim=4, hidden_dim=16).to(device)
        checkpoint = torch.load(full_model_path, map_location=device, weights_only=True)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        # Print a clear tag for the aggregator to know we are ready
        print("✅ GNN Model Loaded Successfully.")
        return model, device

    except Exception as e:
        print(f"❌ Failed to load model architecture: {e}")
        return None, None

def get_risk_score(model, device, url, builder):
    print(f"   [+] Scanning infrastructure for: {url} ...")

    try:
        data = builder.get_features(url)
        print(f"   [+] Graph Built: {data.num_nodes} nodes (URL+Domain+IP+Subdomains).")
        data = data.to(device)
    except Exception as e:
        return None, f"Error parsing URL: {str(e)}"

    with torch.no_grad():
        logits = model(data)
        probability = torch.sigmoid(logits).item()

    return int(probability * 100), None

# --- 3. THE RUN FUNCTION (Automation Friendly) ---
def run():
    print("="*40)
    print("          DEFENDER V4: Arachne")
    print("="*40)

    # 1. Load Model
    model, device = load_brain()

    if model is None:
        print("⚠️  System operating in DIAGNOSTIC MODE (No Neural Network)")
        return

    try:
        builder = UrlGraphBuilder()
    except:
        print("❌ Error: Could not initialize Graph Builder.")
        return

    # 2. Input Handling (Robust)
    try:
        # We read input, but we handle the case where input might be empty
        if len(sys.argv) > 1:
            url = sys.argv[1] # Allow passing URL as argument
        else:
            url = input("\n>> Enter URL to scan: ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    # 3. Process Logic
    if not url or url.lower() in ['exit', 'quit', 'q']:
        return

    # 4. Execute Scan
    score, error = get_risk_score(model, device, url, builder)

    if error:
        print(f"   [!] {error}")
    else:
        # --- CRITICAL: OUTPUT FORMAT FOR AGGREGATOR ---
        # The Aggregator regex looks for: "Risk Score: XX/100"
        if score < 50: color = ""
        elif score < 75: color = "⚠️  "
        else: color = "❌ "

        print(f"   Result: {color}Risk Score: {score}/100")
        print("-" * 30)

    # 5. NO PAUSE
    # We explicitly REMOVED the "Press Enter" input here to prevent the EOFError

if __name__ == "__main__":
    run()
