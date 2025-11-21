import torch
# Import the ensemble predictor
from ai_engines.defender.models.ensemble_predictor import EnsemblePredictor
def dummy_image_tensor():
    """
    Returns a random 3x224x224 tensor to simulate
    a webpage screenshot or HTML-rendered image.
    """
    return torch.rand((1, 3, 224, 224))
def dummy_graph_data():
    """
    Simulates URL/DOM/redirect graph input.
    Replace with real graph objects later.
    """
    return {
        "nodes": ["example.com", "login.php", "cdn.com/script.js"],
        "edges": [
            (0, 1), (1, 2)
        ],
        "features": torch.rand((3, 16))  # 3 nodes, 16-dim features
    }
def main():
    # -------------------------------
    # Load the predictor
    # -------------------------------
    predictor = EnsemblePredictor(
        nlp_weights_path="nlp_model.pth",      # change if needed
        cnn_weights_path="cnn_model.pth",
        gnn_weights_path="gnn_model.pth"
    )
    # -------------------------------
    # Dummy test input
    # -------------------------------
    test_text = """
    Your account has been restricted. Please verify your information
    immediately by clicking the secure link below.
    """
    image_tensor = dummy_image_tensor()
    graph_data = dummy_graph_data()
    # -------------------------------
    # Predict
    # -------------------------------
    output = predictor.predict(
        text=test_text,
        html_image_tensor=image_tensor,
        graph_data=graph_data
    )
    print("\n--- ENSEMBLE TEST OUTPUT ---")
    print(f"NLP Probability : {output['nlp_prob']}")
    print(f"CNN Probability : {output['cnn_prob']}")
    print(f"GNN Probability : {output['gnn_prob']}")
    print(f"Final Combined  : {output['final_score']}")
    print("--------------------------------\n")
if __name__ == "__main__":
    main()
