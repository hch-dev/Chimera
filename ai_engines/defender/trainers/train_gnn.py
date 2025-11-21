# train_gnn.py
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
import pickle
# ==================================================
# CONFIG
# ==================================================
MODEL_PATH = "models/gnn_model.pt"
GRAPH_DATA_PATH = "models/gnn_graph_encoder.pkl"   # saves any encoder/processing objs
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
# ==================================================
# 1. CREATE URL/DOM/REDIRECT GRAPHS
# ==================================================
def load_raw_dataset():
    """
    Replace with real dataset loader.
    Should return: list_of_graphs, list_of_labels
    """
    # ---- Dummy example ----
    # Graph 1 (phishing)
    edge_index_1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x1 = torch.tensor([[1], [0], [1]], dtype=torch.float)
    g1 = Data(x=x1, edge_index=edge_index_1)
    # Graph 2 (benign)
    edge_index_2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x2 = torch.tensor([[0], [1]], dtype=torch.float)
    g2 = Data(x=x2, edge_index=edge_index_2)
    return [g1, g2], [1, 0]
# ==================================================
# 2. PREPROCESS GRAPHS
# ==================================================
def preprocess_graphs(graphs):
    """
    Preprocessing hook if needed (encoding DOM tags, URL tokens, redirect hops).
    For now, direct pass-through.
    """
    # Save encoder (if any) — right now empty placeholder
    with open(GRAPH_DATA_PATH, "wb") as f:
        pickle.dump({"encoder": None}, f)
    return graphs
# ==================================================
# 3. BUILD GNN MODEL
# ==================================================
class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin1 = torch.nn.Linear(hidden_dim, 16)
        self.lin2 = torch.nn.Linear(16, 1)  # Output: probability
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))  # probability output
        return x.squeeze()
# ==================================================
# 4. TRAINING LOOP
# ==================================================
def train():
    # Load data
    raw_graphs, labels = load_raw_dataset()
    # Preprocess graphs
    graphs = preprocess_graphs(raw_graphs)
    # Split dataset
    train_graphs, test_graphs, train_labels, test_labels = train_test_split(
        graphs, labels, test_size=0.2, random_state=42
    )
    # Attach labels
    for g, y in zip(train_graphs, train_labels):
        g.y = torch.tensor([y], dtype=torch.float)
    for g, y in zip(test_graphs, test_labels):
        g.y = torch.tensor([y], dtype=torch.float)
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)
    # Init model
    model = GNNClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # ---- Training ----
    print("\nTraining GNN Model...\n")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.binary_cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")
    # ---- Evaluation ----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            preds = (model(batch) > 0.5).float()
            correct += (preds == batch.y).sum().item()
            total += len(batch.y)
    print(f"\nGNN Test Accuracy: {correct / total:.4f}")
    # ---- Save model ----
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    train()
