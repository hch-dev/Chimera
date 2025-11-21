import torch
import torch.nn.functional as F

# --- FIXED IMPORTS ---
# We use the actual filenames (nlp_analysis) and actual class names (Detector)
from .nlp_analysis import NLPPhishingDetector
from .cnn_analysis import CNNPhishingDetector
from .graph_analysis import GNNPhishingDetector

class EnsemblePredictor:
    """
    Unified phishing probability predictor using NLP + CNN + GNN.
    """
    def __init__(
        self,
        nlp_weights_path: str,
        cnn_weights_path: str,
        gnn_weights_path: str,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Ensemble] Loading models on {self.device}...")

        # ---------------------------
        #  Load NLP model
        # ---------------------------
        self.nlp_model = NLPPhishingDetector().to(self.device)
        # strict=False allows loading even if slight layer mismatches occur during dev
        self.nlp_model.load_state_dict(torch.load(nlp_weights_path, map_location=self.device))
        self.nlp_model.eval()

        # ---------------------------
        #  Load CNN model
        # ---------------------------
        self.cnn_model = CNNPhishingDetector().to(self.device)
        self.cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location=self.device))
        self.cnn_model.eval()

        # ---------------------------
        #  Load GNN model
        # ---------------------------
        self.gnn_model = GNNPhishingDetector().to(self.device)
        self.gnn_model.load_state_dict(torch.load(gnn_weights_path, map_location=self.device))
        self.gnn_model.eval()

        # ---------------------------
        #  Model Weights (Voting Preference)
        # ---------------------------
        self.W_NLP = 0.15
        self.W_CNN = 0.40
        self.W_GNN = 0.45

    # ------------------------------------------------------
    # Preprocessing stubs
    # ------------------------------------------------------
    def preprocess_text(self, text: str):
        # In a real app, this would truncate/clean text
        return text

    def preprocess_image(self, html_or_image):
        if torch.is_tensor(html_or_image):
            return html_or_image.to(self.device)
        return None

    def preprocess_graph(self, graph_data):
        return graph_data

    # ------------------------------------------------------
    # Prediction function
    # ------------------------------------------------------
    @torch.no_grad()
    def predict(self, text: str, html_image_tensor, graph_data):
        """
        Returns ONLY the numerical phishing probability.
        """
        # ---- NLP ----
        clean_text = self.preprocess_text(text)
        nlp_prob = self.nlp_model(clean_text).item()

        # ---- CNN ----
        cnn_tensor = self.preprocess_image(html_image_tensor)
        cnn_prob = self.cnn_model(cnn_tensor).item()

        # ---- GNN ----
        # Note: GNN model expects a graph object, make sure graph_data fits
        gnn_prob = self.gnn_model(graph_data).item()

        # ---- Weighted Final Score ----
        final_score = (
            self.W_NLP * nlp_prob +
            self.W_CNN * cnn_prob +
            self.W_GNN * gnn_prob
        )

        return {
            "nlp_prob": round(nlp_prob, 4),
            "cnn_prob": round(cnn_prob, 4),
            "gnn_prob": round(gnn_prob, 4),
            "final_score": round(final_score, 4)
        }