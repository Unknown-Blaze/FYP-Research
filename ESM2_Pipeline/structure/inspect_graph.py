import torch
from pathlib import Path

# CONFIG
GRAPH_DIR = Path("data/graphs/threaded")

def main():
    # 1. Check if files exist
    mhc_files = list((GRAPH_DIR / "mhc").glob("*.pt"))
    pep_files = list((GRAPH_DIR / "peptide").glob("*.pt"))
    
    if not mhc_files:
        print("❌ No MHC graphs found! Did the generator run?")
        return

    print(f"✅ Found {len(mhc_files)} MHC graphs and {len(pep_files)} Peptide graphs.")

    # 2. Inspect the first MHC graph
    print("\n--- Inspecting First MHC Graph ---")
    mhc_data = torch.load(mhc_files[0])
    print(f"File: {mhc_files[0].name}")
    print(f"Node Features (X) Shape: {mhc_data.x.shape}  <-- Should be (NumResidues, 22)")
    print(f"Positions (Pos) Shape:   {mhc_data.pos.shape} <-- Should be (NumResidues, 3)")
    print(f"Edges (Edge Index):      {mhc_data.edge_index.shape}")
    
    # 3. Inspect the first Peptide graph
    print("\n--- Inspecting First Peptide Graph ---")
    pep_data = torch.load(pep_files[0])
    print(f"File: {pep_files[0].name}")
    print(f"Node Features (X) Shape: {pep_data.x.shape}  <-- Should be (9, 22)")
    print(f"Target (y):              {pep_data.y}         <-- Should be a pIC50 float")

    print("\n✅ READY FOR GNN TRAINING")

if __name__ == "__main__":
    main()