"""
MHC-Peptide Template Threader (Research Grade)
Implements Rigid Backbone Homology Modeling with Sequence Identity Tracking.
"""
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import os
import difflib # Essential for Identity calculation
from torch_cluster import radius_graph

warnings.filterwarnings("ignore")

# CONFIG
TEMPLATE_DIR = "templates"
# Ideally, these PDBs cover different MHC Supertypes (A02, A24, B07, etc.)
TEMPLATE_FILES = [f"template_{t.lower()}.pdb" for t in ["1HHI", "1A1M", "3VCL", "1XR8"]]
OUT_DIR = Path("data/graphs/ensemble")

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "mhc").mkdir(exist_ok=True)

# CHEMICAL DICTIONARIES (Standard Hydrophobicity Scales)
AA_PROPERTIES = {
    'ALA': [0, 0, 0, 0.2], 'ARG': [1, 1, 0, 0.8], 'ASN': [0, 1, 0, 0.5],
    'ASP': [-1, 1, 0, 0.5], 'CYS': [0, 0, 0, 0.4], 'GLN': [0, 1, 0, 0.6],
    'GLU': [-1, 1, 0, 0.6], 'GLY': [0, 0, 0, 0.0], 'HIS': [0.5, 1, 1, 0.6],
    'ILE': [0, 0, 0, 0.7], 'LEU': [0, 0, 0, 0.7], 'LYS': [1, 1, 0, 0.7],
    'MET': [0, 0, 0, 0.7], 'PHE': [0, 0, 1, 0.8], 'PRO': [0, 0, 0, 0.4],
    'SER': [0, 1, 0, 0.3], 'THR': [0, 1, 0, 0.4], 'TRP': [0, 1, 1, 1.0],
    'TYR': [0, 1, 1, 0.9], 'VAL': [0, 0, 0, 0.5],
}
HYDROPHOBICITY = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}

# --- HELPER FUNCTIONS ---

def rbf_expansion(distances, num_centers=16, cutoff=10.0):
    centers = torch.linspace(0, cutoff, num_centers, device=distances.device)
    width = centers[1] - centers[0]
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / width)

def get_physico_chemical_features(aa_char):
    # Standard mapping handling unknown AAs as Alanine
    three_letter_map = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS',
                        'Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE',
                        'L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO',
                        'S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    res = three_letter_map.get(aa_char, 'ALA')
    props = np.array(AA_PROPERTIES.get(res, [0,0,0,0]), dtype=np.float32)
    hyd = np.array([HYDROPHOBICITY.get(res, 0.0) / 4.5], dtype=np.float32)
    return np.concatenate([props, hyd, [1.0]])

def load_template_data(pdb_path):
    """
    Parses PDB to extract BOTH 3D Coordinates AND the original Amino Acid Sequence.
    This allows us to calculate how 'similar' our target is to this template.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", pdb_path)
    
    mhc_data = {'coords': [], 'seq': ""}
    pep_data = {'coords': [], 'seq': ""}
    
    print(f"Scanning {os.path.basename(pdb_path)}...")
    for chain in structure[0]:
        # Filter for Residues only (ignore water, ligands)
        residues = [res for res in chain if 'CA' in res and res.id[0] == ' ']
        length = len(residues)
        
        coords = [res['CA'].get_coord() for res in residues]
        # Biopython: Convert 3-letter resname to 1-letter code
        sequence = "".join([seq1(res.get_resname()) for res in residues])

        if length > 150: # MHC Alpha Chain
            print(f"  -> Found MHC Chain: {length} res")
            mhc_data['coords'] = np.array(coords)
            mhc_data['seq'] = sequence
        elif 8 <= length <= 15: # Peptide
            print(f"  -> Found Peptide Chain: {length} res")
            pep_data['coords'] = np.array(coords)
            pep_data['seq'] = sequence
            
    return mhc_data, pep_data

def create_unified_interaction_graph(mhc_seq, pep_seq, mhc_template, pep_template, ic50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. SAFE LENGTH HANDLING ---
    # We thread the target sequence onto the template backbone.
    # If target is longer, we truncate (Approximation).
    # If target is shorter, we truncate the template.
    
    # MHC
    limit_mhc = min(len(mhc_seq), len(mhc_template['coords']))
    mhc_seq_trunc = mhc_seq[:limit_mhc]
    mhc_pos_trunc = mhc_template['coords'][:limit_mhc]

    # Peptide
    limit_pep = min(len(pep_seq), len(pep_template['coords']))
    pep_seq_trunc = pep_seq[:limit_pep]
    pep_pos_trunc = pep_template['coords'][:limit_pep]

    # --- 2. CALCULATE SIMILARITY (Research Grade) ---
    # Using SequenceMatcher to get % Identity (0.0 to 1.0)
    # This tells the model: "How confident should I be in this structure?"
    if mhc_template['seq']:
        sim_mhc = difflib.SequenceMatcher(None, mhc_seq, mhc_template['seq'], autojunk=False).ratio()
    else:
        sim_mhc = 0.0
    # ------------------------------------------------

    # 3. Build Features (Target Sequence + Chemical Props)
    mhc_feats = [get_physico_chemical_features(aa) for aa in mhc_seq_trunc]
    pep_feats = [get_physico_chemical_features(aa) for aa in pep_seq_trunc]
    x = torch.tensor(np.array(mhc_feats + pep_feats), dtype=torch.float32, device=device)
    
    # 4. Build Geometry (Template Coordinates)
    pos_mhc = torch.tensor(mhc_pos_trunc, dtype=torch.float32, device=device)
    pos_pep = torch.tensor(pep_pos_trunc, dtype=torch.float32, device=device)
    pos = torch.cat([pos_mhc, pos_pep], dim=0)
    
    # 5. Create Masks (MHC vs Peptide)
    batch_mask = torch.cat([
        torch.zeros(len(pos_mhc), device=device), 
        torch.ones(len(pos_pep), device=device)
    ], dim=0)

    # 6. Generate Graph Edges (Radius Graph)
    # Connects atoms within 10 Angstroms
    edge_index = radius_graph(pos, r=10.0, loop=False)
    row, col = edge_index
    dist = (pos[row] - pos[col]).norm(dim=-1)
    edge_attr = rbf_expansion(dist)

    pIC50 = -np.log10(ic50 * 1e-9 + 1e-12)

    # 7. Save Data
    data = Data(
        x=x.cpu(), 
        pos=pos.cpu(), 
        edge_index=edge_index.cpu(), 
        edge_attr=edge_attr.cpu(), 
        chain_mask=batch_mask.cpu().bool(), # Force Boolean for PyTorch Geometric
        y=torch.tensor([pIC50]),
        identity=torch.tensor([sim_mhc], dtype=torch.float32) # Storing the score!
    )
    
    return data

def main():
    # 1. Load Templates
    templates_cache = []
    print("Loading Templates and Extracting Sequences...")
    for t_file in TEMPLATE_FILES:
        path = os.path.join(TEMPLATE_DIR, t_file)
        if os.path.exists(path):
            mhc_data, pep_data = load_template_data(path)
            if len(mhc_data['coords']) > 0:
                templates_cache.append((mhc_data, pep_data))
    
    if not templates_cache:
        print("CRITICAL: No valid templates loaded.")
        return

    # 2. Process CSV
    df = pd.read_csv("../iedb_dataset.csv").dropna(subset=['mhc_sequence', 'peptide', 'binding_affinity_nM'])
    
    # --- NEW LIST TO STORE LOGS ---
    similarity_logs = []
    # ------------------------------

    print(f"Threading {len(df)} samples against {len(templates_cache)} templates...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Store scores for this specific row
        row_scores = {'original_index': idx, 'mhc_sequence': row['mhc_sequence']}
        
        for t_idx, (mhc_template, pep_template) in enumerate(templates_cache):
            try:
                # Returns ONE unified object
                unified_graph = create_unified_interaction_graph(
                    row['mhc_sequence'].upper(), 
                    row['peptide'].upper(), 
                    mhc_template, 
                    pep_template, 
                    row['binding_affinity_nM']
                )
                
                # Save Graph
                torch.save(unified_graph, OUT_DIR / "mhc" / f"{idx}_T{t_idx}.pt")
                
                # --- LOG THE SCORE ---
                # Retrieve the score we just calculated inside the function
                score = float(unified_graph.identity)
                row_scores[f'T{t_idx}_identity'] = score
                # ---------------------
                
            except Exception as e:
                row_scores[f'T{t_idx}_identity'] = 0.0 # Log failure as 0

        similarity_logs.append(row_scores)

        # Optional: Save every 10,000 rows to avoid memory crash
        if len(similarity_logs) % 10000 == 0:
             pd.DataFrame(similarity_logs).to_csv(OUT_DIR / "similarity_metadata_partial.csv", index=False)

    # --- SAVE FINAL CSV ---
    metadata_df = pd.DataFrame(similarity_logs)
    metadata_df.to_csv(OUT_DIR / "similarity_metadata.csv", index=False)
    print(f"Saved similarity logs to {OUT_DIR / 'similarity_metadata.csv'}")

if __name__ == "__main__":
    main()