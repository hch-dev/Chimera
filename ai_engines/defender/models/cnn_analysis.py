import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class CNNPhishingDetector(nn.Module);
    """
    CNN-based phishing classifier using a modified ResNet18.
    Designed for analyzing webpage screenshots or visual HTML renderings.

    Outputs:
        - Embedding vector for final fusion (return_embedding=True)
        - Phishing probability (return_embedding=False)
    """

    # noinspection PyTypeChecker
    def __init__(self, embedding_dim=256):
        super().__init__()

        # ------------------------------
        # 1. Load Pretrained ResNet18
        # ------------------------------
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remove the final classifier layer
        cnn_output_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # ------------------------------
        # 2. Projection layer for embeddings
        # ------------------------------
        self.embedding_head = nn.Sequential(
            nn.Linear(cnn_output_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ------------------------------
        # 3. Probabilistic phishing classifier
        # ------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # ------------------------------
        # 4. Standard image preprocessing
        # ------------------------------
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # ---------------------------------------------------
    # IMAGE PREPROCESSING
    # ---------------------------------------------------

    def prepare_image(self, image_input):
        """
        Accepts:
            - Path to an image
            - PIL Image object

        Returns: transformed tensor of shape (1, 3, 224, 224)
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL Image.")

        img_tensor = self.transform(image).unsqueeze(0)  # (1, 3, 224, 224)
        return img_tensor

    # ---------------------------------------------------
    # FORWARD PASS
    # ---------------------------------------------------

    def forward(self, image_input, return_embedding=False):
        """
        image_input: path to image or PIL image object
        return_embedding: if True, return only the embedding for fusion model
        """

        x = self.prepare_image(image_input)
        features = self.cnn(x)  # (1, 512)
        embedding = self.embedding_head(features)  # (1, embedding_dim)

        if return_embedding:
            return embedding

        prob = self.classifier(embedding)  # phishing probability
        return prob

    # ---------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))
        return self
