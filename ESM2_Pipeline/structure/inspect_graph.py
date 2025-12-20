import torch
import os
from torch_geometric.data import Data

# Paths to your output
mhc_path = "data/graphs/mhc"
pep_path = "data/graphs/peptide"

def check_graph(folder, name):
    files = [f for f in os.listdir(folder) if f.endswith('.pt')]
    if not files:
        print(f"No .pt files found in {folder}")
        return

    # Load the first file found
    file_path = os.path.join(folder, files[0])
    data = torch.load(file_path)
    
    print(f"\n--- Inspecting {name} Graph: {files[0]} ---")
    print(f"Keys: {data.keys}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    
    # Check for specific features
    if hasattr(data, 'x') and data.x is not None:
        print(f"Node Features (x) shape: {data.x.shape} (Num Nodes x Features)")
    else:
        print("WARNING: No node features ('x') found!")

    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print(f"Edge Index shape: {data.edge_index.shape}")
        
    if hasattr(data, 'pos') and data.pos is not None:
        print(f"3D Coordinates (pos) shape: {data.pos.shape}")

# Run checks
check_graph(mhc_path, "MHC")
check_graph(pep_path, "Peptide")