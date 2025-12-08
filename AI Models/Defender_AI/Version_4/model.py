import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class PhishingGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PhishingGNN, self).__init__()

        # --- Layer 1: The "Conversation" Starter ---
        # Takes the raw 4 features and projects them into a higher
        # dimensional space (hidden_dim) to find hidden patterns.
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # --- Layer 2: Deepening the Understanding ---
        # Refines the features based on neighbors' information.
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # --- Layer 3: The Decision Maker ---
        # A simple Linear layer that condenses the graph memory
        # into a single output score.
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. First Graph Convolution
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Activation function (adds non-linearity)
        x = F.dropout(x, p=0.2, training=self.training) # Prevents overfitting

        # 2. Second Graph Convolution
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 3. Pooling (Readout Layer)
        # We have 4 nodes, but we need 1 prediction for the whole graph.
        # global_mean_pool calculates the average of all node features.
        # It uses 'batch' to handle multiple graphs at once during training.
        x = global_mean_pool(x, batch)

        # 4. Final Classification
        x = self.classifier(x)

        # We return the raw output (logits).
        # We will apply Sigmoid in the training/inference loop.
        return x
