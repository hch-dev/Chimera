import os
import json
from typing import Any, Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
# Try to import torch_geometric for a proper GNN implementation.
# If unavailable, we'll fallback to networkx-based aggregation + MLP.
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, global_mean_pool
    HAS_PYTORCH_GEOMETRIC = True
except Exception:
    HAS_PYTORCH_GEOMETRIC = False
# networkx fallback
try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False
# ---------------------------
# Utility: Graph builders
# ---------------------------
def _adjlist_to_nx(adj: Dict[str, List[str]]) -> "nx.Graph":
    """
    Convert adjacency dictionary {node: [neigh, ...], ...} to a networkx DiGraph.
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx not installed; please install networkx for graph utilities.")
    G = nx.DiGraph()
    for u, vs in adj.items():
        G.add_node(u)
        for v in vs:
            G.add_edge(u, v)
    return G
def _redirects_to_adjlist(redirects: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Convert list of redirects into adjacency dict.
    Example: [("a.com","b.com"), ("b.com","c.com")] -> {"a.com":["b.com"], "b.com":["c.com"], "c.com":[]}
    """
    adj = {}
    for a, b in redirects:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, [])  # ensure node present
    return adj
def load_adj_from_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    raise ValueError("JSON file must contain an adjacency dictionary {node: [neighs, ...], ...}")
# ---------------------------
# Helper: node feature extractor
# ---------------------------
def _compute_simple_node_features(G) -> torch.Tensor:
    """
    Given a networkx graph, compute a small vector of node features per node:
    - out-degree
    - in-degree
    - clustering coefficient (0 if DiGraph not supported)
    - pagerank score (normalized)
    Return: tensor shape (num_nodes, feat_dim)
    """
    if not HAS_NETWORKX:
        raise ImportError("networkx required for fallback feature extraction.")
    nodes = list(G.nodes())
    n = len(nodes)
    out_deg = torch.tensor([G.out_degree(n0) if G.is_directed() else G.degree(n0) for n0 in nodes], dtype=torch.float32)
    in_deg = torch.tensor([G.in_degree(n0) if G.is_directed() else G.degree(n0) for n0 in nodes], dtype=torch.float32)
    try:
        pr = nx.pagerank(G.to_undirected()) if n > 0 else {}
        pagerank = torch.tensor([pr.get(n0, 0.0) for n0 in nodes], dtype=torch.float32)
    except Exception:
        pagerank = torch.zeros(n, dtype=torch.float32)
    # clustering coeff (works only for undirected); fallback 0
    try:
        cluster = nx.clustering(G.to_undirected()) if n > 0 else {}
        clustering = torch.tensor([cluster.get(n0, 0.0) for n0 in nodes], dtype=torch.float32)
    except Exception:
        clustering = torch.zeros(n, dtype=torch.float32)
    # stack features: shape (n, 4)
    feats = torch.stack([out_deg, in_deg, pagerank, clustering], dim=1)
    # normalize roughly
    if feats.numel() > 0:
        feats = (feats - feats.mean(dim=0, keepdim=True)) / (feats.std(dim=0, keepdim=True) + 1e-6)
    return feats
# ---------------------------
# GNN Model (PyG) and Fallback
# ---------------------------
class GNNPhishingDetector(nn.Module):
    """
    GNN-based phishing detector.
    Usage:
        model = GNNPhishingDetector(embedding_dim=256)
        prob = model(graph_input)                # probability (tensor 1x1)
        embed = model(graph_input, return_embedding=True)  # embedding tensor (1, embedding_dim)
    graph_input can be:
        - networkx.Graph or DiGraph
        - adjacency dict: {node: [neighs, ...], ...}
        - list of redirects: [("a.com","b.com"), ...]
        - path to JSON file containing adjacency dict
    """
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 128, device: Optional[str] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if HAS_PYTORCH_GEOMETRIC:
            # A 2-layer GraphSAGE encoder + global mean pool
            self.conv1 = SAGEConv(in_channels=4, out_channels=hidden_dim)  # node feature dim = 4 in our extractor
            self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
            self.pool = global_mean_pool
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        else:
            # Fallback: simple MLP that operates on aggregated node features
            # We will compute per-node features (4-d) and aggregate (mean, max, sum) to form graph vector
            agg_dim = 4 * 3  # mean,max,sum of 4-d node features
            self.fallback_mlp = nn.Sequential(
                nn.Linear(agg_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, embedding_dim),
                nn.ReLU(),
            )
        # Classification head (same interface as CNN/NLP)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.to(self.device)
    # ---------------------------
    # Graph preparation
    # ---------------------------
    def _prepare_input(self, graph_input: Union[str, Dict[str, List[str]], List[Tuple[str, str]], Any]):
        """
        Returns either a torch_geometric.data.Data object (if PyG present),
        or (networkx.Graph, node_features_tensor) tuple for fallback.
        """
        # If string pointing to json
        if isinstance(graph_input, str) and os.path.exists(graph_input):
            # try load as adjacency JSON
            if graph_input.lower().endswith(".json"):
                adj = load_adj_from_json(graph_input)
                G = _adjlist_to_nx(adj)
            else:
                # Not JSON; user may pass a path to other formats
                raise ValueError(f"Unsupported file type for graph_input: {graph_input}")
        elif isinstance(graph_input, dict):
            G = _adjlist_to_nx(graph_input)
        elif isinstance(graph_input, list):
            # list of redirects or edges
            adj = _redirects_to_adjlist(graph_input)
            G = _adjlist_to_nx(adj)
        else:
            # possibility: user already passed a networkx Graph
            if HAS_NETWORKX and isinstance(graph_input, (nx.Graph, nx.DiGraph)):
                G = graph_input
            else:
                raise ValueError("Unsupported graph_input type. Provide adjacency dict, list of redirects, path to JSON, or networkx Graph.")
        # compute node features (tensor n x 4)
        node_feats = _compute_simple_node_features(G)  # shape (n, 4)
        if HAS_PYTORCH_GEOMETRIC:
            # build edge_index
            nodes = list(G.nodes())
            idx_map = {n: i for i, n in enumerate(nodes)}
            edges = []
            for u, v in G.edges():
                edges.append((idx_map[u], idx_map[v]))
            if len(edges) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
            x = node_feats.to(self.device)
            data = Data(x=x, edge_index=edge_index.to(self.device))
            return data
        else:
            return G, node_feats.to(self.device)
    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, graph_input: Union[str, Dict[str, List[str]], List[Tuple[str, str]], Any], return_embedding: bool = False):
        """
        graph_input => see _prepare_input
        return_embedding => if True, return graph-level embedding (1, embedding_dim)
        otherwise return phishing probability tensor (1,1)
        """
        self.eval()  # ensure eval mode for deterministic pooling (user can switch .train() externally)
        prepared = self._prepare_input(graph_input)
        if HAS_PYTORCH_GEOMETRIC:
            data = prepared
            x, edge_index = data.x, data.edge_index
            # A small GNN with two SAGE layers
            h = F.relu(self.conv1(x, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            # global mean pool: treat batch=single graph -> batch tensor zeros
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            pooled = self.pool(h, batch)  # shape (1, hidden_dim)
            embedding = self.proj(pooled)  # (1, embedding_dim)
        else:
            G, node_feats = prepared  # node_feats: (n,4)
            if node_feats.size(0) == 0:
                # empty graph → zero embedding
                embedding = torch.zeros((1, self.embedding_dim), device=self.device)
            else:
                mean = node_feats.mean(dim=0)
                mx, _ = node_feats.max(dim=0)
                sm = node_feats.sum(dim=0)
                agg = torch.cat([mean, mx, sm], dim=0).unsqueeze(0)  # (1, agg_dim)
                embedding = self.fallback_mlp(agg.to(self.device))  # (1, embedding_dim)
        if return_embedding:
            return embedding  # shape (1, embedding_dim)
        prob = self.classifier(embedding)  # (1,1) sigmoid output
        return prob
    # ---------------------------
    # Save / Load
    # ---------------------------
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    def load(self, path: str, map_location: Optional[str] = None):
        if map_location is None:
            map_location = self.device
        self.load_state_dict(torch.load(path, map_location=map_location))
        self.to(self.device)
        return self
# ---------------------------
# Small example helpers
# ---------------------------
def example_redirects():
    return [
        ("login.example.com", "secure.example.com"),
        ("secure.example.com", "bank.example.com"),
        ("phishy.example", "bank.example.com")
    ]
if __name__ == "__main__":
    # quick sanity test (works in fallback mode if PyG not available)
    model = GNNPhishingDetector()
    redirects = example_redirects()
    prob = model(redirects)  # tensor
    embed = model(redirects, return_embedding=True)
    print("prob:", prob.item())
    print("embed shape:", embed.shape)
