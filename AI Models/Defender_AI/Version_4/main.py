import sys
import os

# --- 1. SAFE IMPORTS ---
try:
    import torch
    try:
        from graph_builder import UrlGraphBuilder
        from model import PhishingGNN
    except ImportError:
        print("⚠️  Warning: 'graph_builder.py' or 'model.py' not found.")
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
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("   -> You need to run 'train.py' first.")
        return None, None

    try:
        model = PhishingGNN(input_dim=4, hidden_dim=16).to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
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

# --- 3. THE RUN FUNCTION (Single Scan Mode) ---
def run():
    print("="*40)
    print("          DEFENDER V4: Arachne")
    print("="*40)

    # 1. Load Model
    model, device = load_brain()
    
    # 2. Safety Checks
    if model is None:
        print("⚠️  System operating in DIAGNOSTIC MODE (No Neural Network)")
        input("\nPress Enter to return to menu...")
        return

    try:
        builder = UrlGraphBuilder()
    except:
        print("❌ Error: Could not initialize Graph Builder.")
        input("\nPress Enter to return to menu...")
        return

    # 3. Single Input
    try:
        url = input("\n>> Enter URL to scan: ").strip()
    except KeyboardInterrupt:
        return

    # 4. Process Logic
    if not url or url.lower() in ['exit', 'quit', 'q']:
        print("Scan cancelled.")
        return

    # 5. Execute Scan
    score, error = get_risk_score(model, device, url, builder)

    if error:
        print(f"   [!] {error}")
    else:
        if score < 50: color = "" 
        elif score < 75: color = "⚠️  " 
        else: color = "❌ " 
        
        print(f"   Result: {color}Risk Score: {score}/100")
        print("-" * 30)

    # 6. Pause so user can read result
    input("\nPress Enter to return to Main Menu...")

if __name__ == "__main__":
    run()