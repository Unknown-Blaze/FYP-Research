"""
MHC-Peptide Template Threader (Auto-Detect Mode)
Reliably finds MHC and Peptide chains based on length.
"""
import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import os
from torch_cluster import radius_graph # Explicit import

# Suppress PDB warnings
warnings.filterwarnings("ignore")

# CONFIG
TEMPLATE_DIR = "templates"
TEMPLATE_FILES = [f"template_{t.lower()}.pdb" for t in ["1HHI", "1A1M", "3VCL", "1XR8"]]
OUT_DIR = Path("data/graphs/ensemble")
INPUT_CSV = "../iedb_dataset.csv"

# Initialize output directories
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "mhc").mkdir(exist_ok=True)
# We don't strictly need the peptide folder anymore, but keeping it doesn't hurt
(OUT_DIR / "peptide").mkdir(exist_ok=True)

# CHEMICAL DICTIONARIES
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
HYDROPHOBICITY = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}

# Updated Feature Dictionary
AA_PROPERTIES = {
    # Charge, Polarity (0/1), Aromaticity (0/1), VanDerWaals Volume (normed)
    'ALA': [0, 0, 0, 0.2], 'ARG': [1, 1, 0, 0.8], 'ASN': [0, 1, 0, 0.5],
    'ASP': [-1, 1, 0, 0.5], 'CYS': [0, 0, 0, 0.4], 'GLN': [0, 1, 0, 0.6],
    'GLU': [-1, 1, 0, 0.6], 'GLY': [0, 0, 0, 0.0], 'HIS': [0.5, 1, 1, 0.6], # Histidine neutral/protonated
    'ILE': [0, 0, 0, 0.7], 'LEU': [0, 0, 0, 0.7], 'LYS': [1, 1, 0, 0.7],
    'MET': [0, 0, 0, 0.7], 'PHE': [0, 0, 1, 0.8], 'PRO': [0, 0, 0, 0.4],
    'SER': [0, 1, 0, 0.3], 'THR': [0, 1, 0, 0.4], 'TRP': [0, 1, 1, 1.0],
    'TYR': [0, 1, 1, 0.9], 'VAL': [0, 0, 0, 0.5],
}

# --- HELPER FUNCTIONS ---

def rbf_expansion(distances, num_centers=16, cutoff=10.0):
    """
    Expands scalar distances into a vector of radial basis functions.
    This helps the GNN learn non-linear distance relationships.
    """
    centers = torch.linspace(0, cutoff, num_centers, device=distances.device)
    width = centers[1] - centers[0]
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / width)

def get_physico_chemical_features(aa_char):
    three_letter_map = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS',
                        'Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE',
                        'L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO',
                        'S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    res = three_letter_map.get(aa_char, 'ALA')
    
    # 1. Physics Props (4 dims)
    props = np.array(AA_PROPERTIES.get(res, [0,0,0,0]), dtype=np.float32)
    
    # 2. Hydrophobicity (1 dim)
    hyd = np.array([HYDROPHOBICITY.get(res, 0.0) / 4.5], dtype=np.float32)
    
    # 3. pLDDT (1 dim) - Keep this, it guides the GNN to trust good regions
    plddt = np.array([1.0], dtype=np.float32) 
    
    return np.concatenate([props, hyd, plddt]) # Total 6 dims (Very compact)

def load_template_backbone(pdb_path):
    """
    Scans chains by length to find MHC and Peptide.
    MHC: > 200 residues
    Peptide: < 20 residues
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", pdb_path)
    
    mhc_coords = []
    pep_coords = []
    
    print("Scanning PDB Chains:")
    for chain in structure[0]:
        # Get all Alpha Carbons
        ca_atoms = [res['CA'].get_coord() for res in chain if 'CA' in res]
        length = len(ca_atoms)
        
        print(f" -> Chain {chain.get_id()}: {length} residues")
        
        if length > 200:
            print("    -> Identified as MHC")
            mhc_coords = ca_atoms
        elif 8 <= length <= 15:
            print("    -> Identified as PEPTIDE")
            pep_coords = ca_atoms
            
    return np.array(mhc_coords), np.array(pep_coords)

def create_unified_interaction_graph(mhc_seq, pep_seq, mhc_pos, pep_pos, ic50):
    # 0. Select GPU (Uses the first available one)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Concat Features & Coordinates (Create on CPU first)
    mhc_feats = [get_physico_chemical_features(aa) for aa in mhc_seq]
    pep_feats = [get_physico_chemical_features(aa) for aa in pep_seq]
    
    # Move to GPU immediately
    x = torch.tensor(np.array(mhc_feats + pep_feats), dtype=torch.float32, device=device)
    
    # 2. Concat Positions & Move to GPU
    pos_mhc = torch.tensor(mhc_pos, dtype=torch.float32, device=device)
    pos_pep = torch.tensor(pep_pos, dtype=torch.float32, device=device)
    pos = torch.cat([pos_mhc, pos_pep], dim=0)
    
    # 3. Create Batch/Mask vector (GPU)
    batch_mask = torch.cat([
        torch.zeros(len(mhc_pos), device=device), 
        torch.ones(len(pep_pos), device=device)
    ], dim=0)

    # 4. Generate Spatial Edges (NOW FAST ON GPU)
    # torch_cluster will uses CUDA kernels here
    edge_index = radius_graph(pos, r=10.0, loop=False)
    
    # 5. Calculate Edge Distances (GPU)
    row, col = edge_index
    dist = (pos[row] - pos[col]).norm(dim=-1)
    
    # Encode distance using RBF (GPU)
    # THIS FUNCTION WAS MISSING IN YOUR PASTE
    edge_attr = rbf_expansion(dist, num_centers=16, cutoff=10.0)

    # Target
    pIC50 = -np.log10(ic50 * 1e-9 + 1e-12)

    # 6. Move everything BACK to CPU for saving to disk
    # (We don't want to save CUDA tensors; they are messy to load later)
    data = Data(
        x=x.cpu(), 
        pos=pos.cpu(), 
        edge_index=edge_index.cpu(), 
        edge_attr=edge_attr.cpu(), 
        chain_mask=batch_mask.cpu(), 
        y=torch.tensor([pIC50])
    )
    
    return data

def main():
    # 1. Load ALL Templates into memory first
    templates_cache = []
    print("Loading Templates...")
    for t_file in TEMPLATE_FILES:
        path = os.path.join(TEMPLATE_DIR, t_file)
        mhc_c, pep_c = load_template_backbone(path)
        templates_cache.append((mhc_c, pep_c))
        print(f"  Loaded {t_file}: MHC={len(mhc_c)}, Pep={len(pep_c)}")

    # 2. Process CSV
    df = pd.read_csv("../iedb_dataset.csv").dropna(subset=['mhc_sequence', 'peptide', 'binding_affinity_nM'])
    
    # Ensure directories exist
    (OUT_DIR / "mhc").mkdir(parents=True, exist_ok=True)

    print(f"Threading {len(df)} samples against {len(templates_cache)} templates...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for t_idx, (mhc_coords, pep_coords) in enumerate(templates_cache):
            try:
                # Returns ONE unified object
                unified_graph = create_unified_interaction_graph(
                    row['mhc_sequence'].upper(), 
                    row['peptide'].upper(), 
                    mhc_coords, 
                    pep_coords, 
                    row['binding_affinity_nM']
                )
                
                # Save into the 'mhc' folder (serving as the main graph folder now)
                torch.save(unified_graph, OUT_DIR / "mhc" / f"{idx}_T{t_idx}.pt")
                
            except Exception as e:
                pass # Skip failed threads

if __name__ == "__main__":
    main()