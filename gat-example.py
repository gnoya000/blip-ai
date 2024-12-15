import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd
import numpy as np

# --- Configuration ---
learning_rate = 0.01
epochs = 200

# --- Read the adjacency matrix from CSV ---
adjacency_df = pd.read_csv('adjacency-matrix.csv')  # shape: (num_edges, 2)
adjacency_df = adjacency_df[0:20000]
# Create an adjacency matrix (symmetric, undirected graph)
num_nodes = adjacency_df['From_Node'].max() + 1  # Assuming node ids start from 0
adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
# Fill the adjacency matrix with ones where there is an edge
for _, row in adjacency_df.iterrows():
    print('row: ',row)
    from_node, to_node = int(row['From_Node']), int(row['To_Node'])
    adjacency_matrix[from_node, to_node] = 1
    adjacency_matrix[to_node, from_node] = 1  # Because the graph is undirected

# --- Read the node features from CSV ---
features_df = pd.read_csv('synthetic_social_network_features.csv')
# Convert the features to a PyTorch tensor
X = torch.tensor(features_df.to_numpy(), dtype=torch.float32)

# --- GAT Layer Implementation ---
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features))  # Weight matrix
        self.a = nn.Parameter(torch.randn(2 * out_features, 1))  # Attention mechanism
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, X, A):
        # Linear Transformation
        Z = X @ self.W  # Shape: (num_nodes, out_features)

        # Compute attention scores for each edge
        e = torch.zeros_like(A)  # Shape: (num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if A[i, j] > 0:  # Only compute for connected nodes
                    z_concat = torch.cat([Z[i], Z[j]])  # Concatenate features
                    e[i, j] = self.leaky_relu(torch.dot(self.a.squeeze(), z_concat))

        # Normalize attention scores with softmax
        alpha = F.softmax(e, dim=1)  # Normalize scores row-wise (node neighbors)

        # Aggregate features
        h = torch.matmul(alpha, Z)  # Weighted sum of neighbors' features
        return h

# --- GAT Model ---
class GAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(GAT, self).__init__()
        self.gat = GraphAttentionLayer(in_features, out_features)

    def forward(self, X, A):
        return self.gat(X, A)

# --- Edge Predictor ---
class EdgePredictor(nn.Module):
    def __init__(self):
        super(EdgePredictor, self).__init__()

    def forward(self, h):
        num_nodes = h.shape[0]
        scores = torch.zeros((num_nodes, num_nodes))  # Initialize similarity scores
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Use dot product for similarity
                    scores[i, j] = torch.dot(h[i], h[j])
        return scores

# --- Training Setup ---
model = GAT(X.shape[1], 2)  # Input features = 33, output features = 2
predictor = EdgePredictor()
optimizer = Adam(list(model.parameters()) + list(predictor.parameters()), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()  # Binary classification for edge prediction

# Positive and Negative Samples
pos_edges = torch.tensor([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)], dtype=torch.long)  # Existing edges
neg_edges = torch.tensor([(0, 2), (1, 3), (3, 4), (2, 5)], dtype=torch.long)  # Non-existing edges

# --- Training Loop ---
for epoch in range(epochs):
    print(f"Epoch: {epoch}")
    print("Model training...")
    model.train()
    print("Zero grad...")
    optimizer.zero_grad()

    # Forward pass through GAT
    h = model(X, adjacency_matrix)

    # Compute scores for all pairs
    print("Predicting scores...")
    scores = predictor(h)

    # Prepare labels and predictions
    labels = torch.cat([
        torch.ones(len(pos_edges)),  # Positive samples
        torch.zeros(len(neg_edges))  # Negative samples
    ])
    preds = torch.cat([
        scores[edge[0], edge[1]].unsqueeze(0) for edge in pos_edges
    ] + [
        scores[edge[0], edge[1]].unsqueeze(0) for edge in neg_edges
    ])

    # Compute loss
    print("Computing loss...")

    loss = criterion(preds, labels)

    # Backward pass and parameter update
    print("Backward propagation and Parameter Update...")
    loss.backward()
    optimizer.step()

    # Log progress
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- Recommendations ---
print("Recommendation Scores (non-existing edges):")
for edge in neg_edges:
    print(f"Edge {tuple(edge.tolist())}: {scores[edge[0], edge[1]].item():.4f}")