#!/usr/bin/env python
"""
Test script to verify structure pipeline end-to-end
"""
import os
import sys
import json
from pathlib import Path

def test_structure_generation():
    """Test if PDB files are generated"""
    print("\n" + "="*60)
    print("TEST 1: Structure Generation")
    print("="*60)
    
    mapping_file = "./structure/data/structures/mapping.json"
    
    if not os.path.exists(mapping_file):
        print("❌ Mapping file not found")
        print(f"   Expected: {mapping_file}")
        print(f"   Run: python run_alphafold.py first")
        return False
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    success_count = len(mapping.get('success', {}))
    failed_count = len(mapping.get('failed', []))
    backend = mapping.get('backend', 'unknown')
    
    print(f"✓ Mapping file found")
    print(f"  Backend: {backend}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    
    # Verify PDB files exist
    pdb_files = mapping.get('success', {})
    missing = []
    
    for idx, pdb_path in list(pdb_files.items())[:5]:  # Check first 5
        if not os.path.exists(pdb_path):
            missing.append((idx, pdb_path))
    
    if missing:
        print(f"❌ Missing PDB files:")
        for idx, path in missing:
            print(f"   [{idx}] {path}")
        return False
    
    print(f"✓ PDB files verified (checked {min(5, len(pdb_files))} samples)")
    
    return True


def test_graph_generation():
    """Test if graphs are generated from PDBs"""
    print("\n" + "="*60)
    print("TEST 2: Graph Generation")
    print("="*60)
    
    mapping_file = "./structure/data/graphs/mapping.json"
    
    if not os.path.exists(mapping_file):
        print("❌ Graph mapping file not found")
        print(f"   Expected: {mapping_file}")
        print(f"   Run: python build_graphs.py")
        return False
    
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    success_count = len(mapping.get('success', {}))
    failed_count = len(mapping.get('failed', []))
    format_type = mapping.get('format', 'unknown')
    
    print(f"✓ Graph mapping file found")
    print(f"  Format: {format_type}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failed_count}")
    
    # Verify graph files exist
    graph_files = mapping.get('success', {})
    missing = []
    
    for idx, graph_path in list(graph_files.items())[:5]:
        if not os.path.exists(graph_path):
            missing.append((idx, graph_path))
        
        # Check if .pt file also exists (PyTorch format)
        pt_path = graph_path.replace('.npz', '.pt')
        if os.path.exists(pt_path):
            print(f"✓ PyTorch format available: {os.path.basename(pt_path)}")
    
    if missing:
        print(f"❌ Missing graph files:")
        for idx, path in missing:
            print(f"   [{idx}] {path}")
        return False
    
    print(f"✓ Graph files verified (checked {min(5, len(graph_files))} samples)")
    
    return True


def test_graph_loading():
    """Test if graphs can be loaded"""
    print("\n" + "="*60)
    print("TEST 3: Graph Loading")
    print("="*60)
    
    try:
        import numpy as np
        print("✓ NumPy available")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    # Find a graph file
    graph_dir = "./structure/data/graphs"
    graph_files = list(Path(graph_dir).glob("graph_*.npz"))
    
    if not graph_files:
        print("❌ No graph files found")
        return False
    
    test_file = graph_files[0]
    print(f"  Testing: {test_file.name}")
    
    # Load numpy graph
    data = np.load(test_file, allow_pickle=True)
    
    print(f"✓ NumPy graph loaded successfully")
    print(f"  Keys: {list(data.keys())}")
    print(f"  Nodes: {data['num_nodes']}")
    print(f"  Edges: {data['edges'].shape}")
    print(f"  Node features: {data['node_features'].shape}")
    print(f"  Positions: {data['positions'].shape}")
    
    # Try PyTorch format if available
    try:
        import torch
        from torch_geometric.data import Data
        
        pt_file = str(test_file).replace('.npz', '.pt')
        if os.path.exists(pt_file):
            graph = torch.load(pt_file)
            print(f"✓ PyTorch Geometric graph loaded")
            print(f"  x (features): {graph.x.shape}")
            print(f"  pos (coords): {graph.pos.shape}")
            print(f"  edge_index: {graph.edge_index.shape}")
            print(f"  edge_attr: {graph.edge_attr.shape}")
    except ImportError:
        print("ℹ PyTorch/PyG not available (optional)")
    
    return True


def test_integration_readiness():
    """Check if ready for sequence+structure integration"""
    print("\n" + "="*60)
    print("TEST 4: Integration Readiness")
    print("="*60)
    
    # Check sequence model checkpoint
    seq_checkpoint = "./sequence/checkpoints/checkpoints_esm2_warmstart/best_model.pt"
    # Note: This might be in different location based on config
    
    # Check if dataset matches
    dataset_file = "./iedb_dataset.csv"
    if not os.path.exists(dataset_file):
        print("❌ Dataset not found")
        return False
    
    print(f"✓ Dataset found: {dataset_file}")
    
    # Check structure outputs
    pdb_mapping = "./structure/data/structures/mapping.json"
    graph_mapping = "./structure/data/graphs/mapping.json"
    
    if not os.path.exists(pdb_mapping):
        print("❌ PDB mapping missing")
        return False
    
    if not os.path.exists(graph_mapping):
        print("❌ Graph mapping missing")
        return False
    
    # Load both mappings
    with open(pdb_mapping, 'r') as f:
        pdb_data = json.load(f)
    
    with open(graph_mapping, 'r') as f:
        graph_data = json.load(f)
    
    pdb_indices = set(pdb_data.get('success', {}).keys())
    graph_indices = set(graph_data.get('success', {}).keys())
    
    # Check consistency
    if pdb_indices != graph_indices:
        print(f"⚠ Warning: PDB and graph indices don't match")
        print(f"  PDB samples: {len(pdb_indices)}")
        print(f"  Graph samples: {len(graph_indices)}")
        print(f"  Missing in graphs: {pdb_indices - graph_indices}")
    else:
        print(f"✓ PDB and graph indices match ({len(pdb_indices)} samples)")
    
    print(f"\n✓ Structure pipeline ready for integration!")
    print(f"  You can now create a multimodal model that combines:")
    print(f"  - Sequence embeddings from ESM2")
    print(f"  - Structure graphs from AlphaFold PDBs")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("STRUCTURE PIPELINE VERIFICATION")
    print("="*60)
    
    tests = [
        ("Structure Generation", test_structure_generation),
        ("Graph Generation", test_graph_generation),
        ("Graph Loading", test_graph_loading),
        ("Integration Readiness", test_integration_readiness),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with error:")
            print(f"   {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Structure pipeline is working correctly.")
        print("\nNext steps:")
        print("1. Create a multimodal dataset class that loads both:")
        print("   - Sequence data (MHC, peptide)")
        print("   - Graph data (from ./structure/data/graphs/)")
        print("2. Create a fusion model that combines:")
        print("   - ESM2 sequence embeddings")
        print("   - GNN structure embeddings")
        print("3. Train the integrated model")
    else:
        print("\n✗ Some tests failed. Fix issues above before integration.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
