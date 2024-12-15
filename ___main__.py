import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from graph_nn import LocationBasedSocialNetworkGAT
from neo4j import GraphDatabase


if(__name__=="__main__"):
    # Neo4j connection
    uri = "bolt://localhost:7687"  # Change this to your Neo4j instance URI
    username = "neo4j"  # Neo4j username
    password = "password"  # Neo4j password
    # Node feature matrix:
    # Each row represents features for a node (e.g., user attributes).


    driver = GraphDatabase.driver(uri, auth=(username, password))

    def get_nodes_and_edges(tx):
    # Query nodes with features (assuming each node has a feature vector)
        result_nodes = tx.run("MATCH (n:User) RETURN id(n) AS node_id, n.features AS features")
        nodes = [(record["node_id"], record["features"]) for record in result_nodes]

        # Query edges
        result_edges = tx.run("MATCH (n1)-[r]->(n2) RETURN id(n1) AS source, id(n2) AS target")
        edges = [(record["source"], record["target"]) for record in result_edges]

        return nodes, edges

    # Retrieve nodes and edges
    with driver.session() as session:
        nodes, edges = session.read_transaction(get_nodes_and_edges)

    # Process nodes to create feature matrix X
    node_id_to_index = {node_id: idx for idx, (node_id, _) in enumerate(nodes)}
    x = torch.tensor([features for _, features in nodes], dtype=torch.float)

    # Process edges to create edge index
    edge_index = [[node_id_to_index[src] for src, _ in edges],
                [node_id_to_index[tgt] for _, tgt in edges]]
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    # Close the driver connection
    driver.close()

    print(data)

    # Initialize the GAT model:
    # - Input feature dimension: 1 (e.g., single feature per user).
    # - Hidden channels: 8 (dimensionality of hidden embeddings).
    # - Output channels: 2 (e.g., final embedding size or prediction classes).
    # - Heads: 2 (multi-head attention).
    model = LocationBasedSocialNetworkGAT(node_input_feature_dimension=33, hidden_channels=64, out_channels=16, heads=4)

    # Initialize an optimizer (Adam) with a small learning rate:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop:
    for epoch in range(1):  
        model.train()  
        optimizer.zero_grad() 

        # Forward pass through the model:
        out = model(data)

        # Define a dummy target (for demonstration):
        # Replace this with actual labels in a real-world scenario.

        ## TODO: FIX
        target = torch.tensor([[1.], [0.], [1.]], dtype=torch.float)  # Dummy labels.

        # Compute the Mean Squared Error (MSE) loss:
        loss = F.mse_loss(out, target)

        # Backward pass (compute gradients) and optimization step:
        loss.backward()
        optimizer.step()

        # Print loss every 20 epochs for monitoring:
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # After training, the output `out` contains the final node embeddings:
    print("Final node embeddings:")
    print(out)

