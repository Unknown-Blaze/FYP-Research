# pdb_to_graph_interaction.py

import torch
import numpy as np
from torch_geometric.data import Data
from pathlib import Path

# --- CHEMICAL KNOWLEDGE BANK ---
# 1. Hydrophobicity (Kyte-Doolittle normalized)
HYDROPHOBICITY = {'ALA': 1.8,'ARG':-4.5,'ASN':-3.5,'ASP':-3.5,'CYS': 2.5,'GLN':-3.5,'GLU':-3.5,'GLY':-0.4,'HIS':-3.2,'ILE': 4.5,'LEU': 3.8,'LYS':-3.9,'MET': 1.9,'PHE': 2.8,'PRO':-1.6,'SER':-0.8,'THR':-0.7,'TRP':-0.9,'TYR':-1.3,'VAL': 4.2}
# 2. Charge
CHARGE = {'ARG': 1.0, 'LYS': 1.0, 'ASP': -1.0, 'GLU': -1.0, 'HIS': 0.1} # His is pH dependent, 0.1 is a soft proxy
# 3. H-Bond Donors/Acceptors (Simplified)
HBOND = {
    'ASN': [1,1], 'GLN': [1,1], 'SER': [1,1], 'THR': [1,1], 'TYR': [1,1], 
    'ASP': [0,1], 'GLU': [0,1], 'LYS': [1,0], 'ARG': [1,0], 'HIS': [1,1]
}

def get_node_features(atom):
    res = atom['res_name']
    
    # Feature 1: One-hot Atom Type (C, N, O, S, Other) - 5 dims
    atom_type = ['C', 'N', 'O', 'S']
    a_onehot = [1 if atom['element'] == a else 0 for a in atom_type]
    if sum(a_onehot) == 0: a_onehot.append(1) # Other
    else: a_onehot.append(0)

    # Feature 2: Physicochemical Properties - 4 dims
    hyd = HYDROPHOBICITY.get(res, 0.0) / 4.5
    chg = CHARGE.get(res, 0.0)
    hb = HBOND.get(res, [0, 0]) # [Donor, Acceptor]

    # Feature 3: Is this node from MHC (0) or Peptide (1)? - 1 dim
    # (This helps the GNN distinguish the two chains)
    chain_tag = [atom['chain_tag']] 

    return np.concatenate([a_onehot, [hyd, chg], hb, chain_tag])

def complex_to_graph(mhc_atoms, pep_atoms, threshold=5.0):
    # Tag atoms to know where they came from
    for a in mhc_atoms: a['chain_tag'] = 0.0
    for a in pep_atoms: a['chain_tag'] = 1.0
    
    all_atoms = mhc_atoms + pep_atoms
    
    # 1. Build Nodes
    node_feats = [get_node_features(a) for a in all_atoms]
    pos = [a['pos'] for a in all_atoms]
    
    x = torch.tensor(np.array(node_feats), dtype=torch.float32)
    pos = torch.tensor(np.array(pos), dtype=torch.float32)

    # 2. Build Edges (The Interaction Logic)
    # We use cdist for efficiency
    from scipy.spatial.distance import cdist
    coords = np.array(pos)
    dists = cdist(coords, coords)
    
    # Create mask: distance < 5A AND distance > 0
    mask = (dists < threshold) & (dists > 1e-6)
    src, dst = np.where(mask)
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # Edge Attributes: [Distance, 1/Distance, Is_Interface_Edge?]
    edge_attrs = []
    for s, d in zip(src, dst):
        d_val = dists[s, d]
        # Check if this edge connects MHC to Peptide
        is_interface = 1.0 if all_atoms[s]['chain_tag'] != all_atoms[d]['chain_tag'] else 0.0
        edge_attrs.append([d_val, 1.0/(d_val+1e-6), is_interface])
        
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)