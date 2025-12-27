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
(OUT_DIR / "peptide").mkdir(exist_ok=True)

# CHEMICAL DICTIONARIES
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
HYDROPHOBICITY = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}

def get_residue_features(aa_char):
    three_letter_map = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    res_name = three_letter_map.get(aa_char, 'ALA')
    
    # 1. Identity (21)
    res_one_hot = np.zeros(21, dtype=np.float32)
    if res_name in AA_TO_IDX: res_one_hot[AA_TO_IDX[res_name]] = 1.0
    else: res_one_hot[-1] = 1.0
    
    # 2. Hydrophobicity (1)
    hyd = [HYDROPHOBICITY.get(res_name, 0.0) / 4.5]
    
    # 3. Confidence (1) -> SET TO 1.0 (TRUSTED CRYSTAL)
    plddt = [1.0] 
    
    return np.concatenate([res_one_hot, hyd, plddt])

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

def create_threaded_graph(mhc_seq, pep_seq, mhc_template_pos, pep_template_pos, ic50):
    # Thread MHC (Truncate or Pad to match template)
    mhc_feats = []
    target_mhc_len = len(mhc_template_pos)
    for i in range(target_mhc_len):
        aa = mhc_seq[i] if i < len(mhc_seq) else 'A'
        mhc_feats.append(get_residue_features(aa))
        
    # Thread Peptide (Truncate or Pad)
    pep_feats = []
    target_pep_len = len(pep_template_pos)
    for i in range(target_pep_len):
        aa = pep_seq[i] if i < len(pep_seq) else 'A'
        pep_feats.append(get_residue_features(aa))
        
    # Build Tensors
    x_mhc = torch.tensor(np.array(mhc_feats), dtype=torch.float32)
    pos_mhc = torch.tensor(mhc_template_pos, dtype=torch.float32)
    x_pep = torch.tensor(np.array(pep_feats), dtype=torch.float32)
    pos_pep = torch.tensor(pep_template_pos, dtype=torch.float32)
    
    # Build Edges
    edge_index_mhc = torch.stack([torch.arange(target_mhc_len-1), torch.arange(1, target_mhc_len)], dim=0)
    edge_index_pep = torch.stack([torch.arange(target_pep_len-1), torch.arange(1, target_pep_len)], dim=0)
    
    # Dummy Edge Attr (Dim 16 for GNN RBF)
    edge_attr_mhc = torch.zeros((edge_index_mhc.size(1), 16), dtype=torch.float32)
    edge_attr_pep = torch.zeros((edge_index_pep.size(1), 16), dtype=torch.float32)

    # Target
    pIC50 = -np.log10(ic50 * 1e-9 + 1e-12)

    mhc_data = Data(x=x_mhc, pos=pos_mhc, edge_index=edge_index_mhc, edge_attr=edge_attr_mhc)
    pep_data = Data(x=x_pep, pos=pos_pep, edge_index=edge_index_pep, edge_attr=edge_attr_pep, y=torch.tensor([pIC50]))
    
    return mhc_data, pep_data

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
    
    (OUT_DIR / "mhc").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "peptide").mkdir(parents=True, exist_ok=True)

    print(f"Threading {len(df)} samples against {len(templates_cache)} templates...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Generate 4 versions of the graph
        for t_idx, (mhc_coords, pep_coords) in enumerate(templates_cache):
            try:
                mhc_g, pep_g = create_threaded_graph(
                    row['mhc_sequence'].upper(), 
                    row['peptide'].upper(), 
                    mhc_coords, 
                    pep_coords, 
                    row['binding_affinity_nM']
                )
                # Save with T{index} suffix
                torch.save(mhc_g, OUT_DIR / "mhc" / f"{idx}_T{t_idx}.pt")
                torch.save(pep_g, OUT_DIR / "peptide" / f"{idx}_T{t_idx}.pt")
            except Exception:
                pass # Skip failed threads

if __name__ == "__main__":
    main()