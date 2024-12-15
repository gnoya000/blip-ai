import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class LocationBasedSocialNetworkGAT(nn.Module):
    def __init__(self, node_input_feature_dimension, hidden_channels, out_channels, heads=2):
        """
        Initializes the Graph Attention Network (GAT) for a location-based social network.

        Parameters:
        - node_input_feature_dimension: Number of input features per node (e.g., user attributes).
        - hidden_channels: Number of hidden units (features) in the intermediate layer.
        - out_channels: Number of output features per node (e.g., embedding size or prediction).
        - heads: Number of attention heads for the GAT layer (default: 1).
        """
        super(LocationBasedSocialNetworkGAT, self).__init__()

        self.conv1 = GATConv(node_input_feature_dimension, hidden_channels, heads=heads)

        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=2)

    def forward(self, data):
        """
        Forward pass through the network.

        Parameters:
        - data: A PyTorch Geometric `Data` object containing graph information.
            - data.x: Node feature matrix of shape [num_nodes, num_features].
            - data.edge_index: Edge indices (connectivity) in COO format.

        Returns:
        - Node embeddings or predictions, with shape [num_nodes, out_channels].
        """
        x, edge_index = data.x, data.edge_index  # Extract node features and edge connectivity.

        x = self.conv1(x, edge_index)

        x = F.relu(x)

        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        return x

# # Create a sample graph for demonstration:
# # - Nodes represent users in a location-based social network.
# # - Edges represent relationships or proximity between users.
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)  # Edge connectivity (COO format).

# # Node feature matrix:
# # Each row represents features for a node (e.g., user attributes).
# x = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float)  # Dummy features for 3 nodes.

# # Create a PyTorch Geometric Data object for the graph:
# data = Data(x=x, edge_index=edge_index)

# # Initialize the GAT model:
# # - Input feature dimension: 1 (e.g., single feature per user).
# # - Hidden channels: 8 (dimensionality of hidden embeddings).
# # - Output channels: 2 (e.g., final embedding size or prediction classes).
# # - Heads: 2 (multi-head attention).
# model = LocationBasedSocialNetworkGAT(node_input_feature_dimension=1, hidden_channels=8, out_channels=2, heads=2)

# # Initialize an optimizer (Adam) with a small learning rate:
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # Training loop:
# for epoch in range(200):  # Train for 200 epochs.
#     model.train()  # Set the model to training mode.
#     optimizer.zero_grad()  # Reset gradients.

#     # Forward pass through the model:
#     # The output `out` contains node embeddings or predictions.
#     out = model(data)

#     # Define a dummy target (for demonstration):
#     # Replace this with actual labels in a real-world scenario.
#     target = torch.tensor([[1.], [0.], [1.]], dtype=torch.float)  # Dummy labels.

#     # Compute the Mean Squared Error (MSE) loss:
#     loss = F.mse_loss(out, target)

#     # Backward pass (compute gradients) and optimization step:
#     loss.backward()
#     optimizer.step()

#     # Print loss every 20 epochs for monitoring:
#     if epoch % 20 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item()}")

# # After training, the output `out` contains the final node embeddings:
# print("Final node embeddings:")
# print(out)

# --- Explanation of Attention ---
# The GATConv layers incorporate attention mechanisms:
# 1. Each node computes attention coefficients for its neighbors based on feature similarity.
# 2. These coefficients determine the importance (weight) of each neighbor during message aggregation.
# 3. Multi-head attention (e.g., `heads=2`) improves robustness by learning independent attention scores.
#
# Advantages of Attention:
# - Enables the model to focus more on important neighbors.
# - Improves performance on graphs with varying node degrees or noisy connections.