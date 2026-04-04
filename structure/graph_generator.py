"""
MHC-Peptide Template Threader (Physics-Enhanced)
FIXED VERSION: Aligns load_template_data with create_unified_interaction_graph
"""
import torch
import math
from Bio.PDB import PDBParser, ShrakeRupley, PPBuilder
from Bio.SeqUtils import seq1
from torch_geometric.data import Data
from torch_cluster import radius_graph
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import warnings
import os
import difflib

warnings.filterwarnings("ignore")

# CONFIG
TEMPLATE_DIR = "templates"
TEMPLATE_FILES = [f"template_{t.lower()}.pdb" for t in ["1HHI", "1A1M", "3VCL", "1XR8"]]
OUT_DIR = Path("data/graphs/ensemble")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "mhc").mkdir(exist_ok=True)

# --- 1. CHEMICAL DICTIONARIES (Target Sequence Features) ---
AA_PROPERTIES = {
    'ALA': [0, 0, 0, 0],   'ARG': [1, 1, 0, 1],   'ASN': [0, 1, 0, 1],
    'ASP': [-1, 1, 0, 1],  'CYS': [0, 0, 0, 0],   'GLN': [0, 1, 0, 1],
    'GLU': [-1, 1, 0, 1],  'GLY': [0, 0, 0, 0],   'HIS': [0.5, 1, 1, 1],
    'ILE': [0, 0, 0, 0],   'LEU': [0, 0, 0, 0],   'LYS': [1, 1, 0, 1],
    'MET': [0, 0, 0, 0],   'PHE': [0, 0, 1, 0],   'PRO': [0, 0, 0, 0],
    'SER': [0, 1, 0, 1],   'THR': [0, 1, 0, 1],   'TRP': [0, 1, 1, 1],
    'TYR': [0, 1, 1, 1],   'VAL': [0, 0, 0, 0],
}
HYDROPHOBICITY = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 
                  'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 
                  'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 
                  'TYR': -1.3, 'VAL': 4.2}

# --- 2. PHYSICS HELPER ---
def compute_template_physics(chain):
    ppb = PPBuilder()
    physics_features = []
    polys = ppb.build_peptides(chain)
    angle_map = {}
    for poly in polys:
        phi_psi = poly.get_phi_psi_list()
        for i, res in enumerate(poly):
            angle_map[res.get_full_id()] = phi_psi[i]

    for res in chain:
        if 'CA' not in res or res.id[0] != ' ': continue
        sasa = res.sasa if hasattr(res, 'sasa') else 0.0
        norm_sasa = min(sasa / 200.0, 1.0)
        phi, psi = angle_map.get(res.get_full_id(), (None, None))
        phi = phi if phi is not None else 0.0
        psi = psi if psi is not None else 0.0
        feats = [norm_sasa, math.sin(phi), math.cos(phi), math.sin(psi), math.cos(psi)]
        physics_features.append(np.array(feats, dtype=np.float32))
    return np.array(physics_features)

def load_template_data(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("template", pdb_path)
    sr = ShrakeRupley()
    try: sr.compute(structure, level="R")
    except: pass

    mhc_data = {'coords_ca': [], 'coords_cb': [], 'seq': "", 'physics': []}
    pep_data = {'coords_ca': [], 'coords_cb': [], 'seq': "", 'physics': []}
    
    print(f"Scanning {os.path.basename(pdb_path)}...")
    for chain in structure[0]:
        residues = [res for res in chain if 'CA' in res and res.id[0] == ' ']
        length = len(residues)
        if length == 0: continue
        coords_ca, coords_cb = [], []
        for res in residues:
            ca_vec = res['CA'].get_vector().get_array()
            coords_ca.append(ca_vec)
            if 'CB' in res: coords_cb.append(res['CB'].get_vector().get_array())
            else: coords_cb.append(ca_vec)
        sequence = "".join([seq1(res.get_resname()) for res in residues])
        phys_feats = compute_template_physics(chain)

        if length > 150:
            print(f"  -> Found MHC Chain: {length} res")
            mhc_data.update({'coords_ca': np.array(coords_ca), 'coords_cb': np.array(coords_cb), 'seq': sequence, 'physics': phys_feats})
        elif 8 <= length <= 15:
            print(f"  -> Found Peptide Chain: {length} res")
            pep_data.update({'coords_ca': np.array(coords_ca), 'coords_cb': np.array(coords_cb), 'seq': sequence, 'physics': phys_feats})
    return mhc_data, pep_data

# --- 3. TARGET SEQ FEATS ---
def get_target_seq_features(aa_char):
    three_letter = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
    res = three_letter.get(aa_char, 'ALA')
    aa_order = sorted(list(AA_PROPERTIES.keys()))
    one_hot = [1.0 if k == res else 0.0 for k in aa_order]
    props = AA_PROPERTIES.get(res, [0,0,0,0])
    hyd = HYDROPHOBICITY.get(res, 0.0) / 4.5
    return np.array(one_hot + props + [hyd], dtype=np.float32)

# --- 4. EDGE PHYSICS ---
def compute_edge_features(pos_ca, pos_cb, edge_index):
    row, col = edge_index
    diff = pos_ca[row] - pos_ca[col]
    dist = diff.norm(dim=-1)
    centers = torch.linspace(0, 10, 16, device=pos_ca.device)
    width = centers[1] - centers[0]
    rbf_dist = torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / width)
    u_sc = pos_cb[row] - pos_ca[row]
    v_sc = pos_cb[col] - pos_ca[col]
    u_norm = torch.nn.functional.normalize(u_sc, dim=-1)
    v_norm = torch.nn.functional.normalize(v_sc, dim=-1)
    orientation = (u_norm * v_norm).sum(dim=-1).unsqueeze(-1)
    return torch.cat([rbf_dist, orientation, dist.unsqueeze(-1)], dim=-1)

# --- 5. MAIN THREADER ---
def create_unified_interaction_graph(mhc_seq, pep_seq, mhc_template, pep_template, ic50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # UPDATED: Use coords_ca instead of coords
    limit_mhc = min(len(mhc_seq), len(mhc_template['coords_ca']))
    limit_pep = min(len(pep_seq), len(pep_template['coords_ca']))
    mhc_seq_t = mhc_seq[:limit_mhc]
    pep_seq_t = pep_seq[:limit_pep]

    mhc_target_feats = [get_target_seq_features(aa) for aa in mhc_seq_t]
    pep_target_feats = [get_target_seq_features(aa) for aa in pep_seq_t]
    mhc_struct_feats = mhc_template['physics'][:limit_mhc]
    pep_struct_feats = pep_template['physics'][:limit_pep]
    
    # Feature Concatenation (Dim 30)
    mhc_final = np.concatenate([mhc_target_feats, mhc_struct_feats], axis=1)
    pep_final = np.concatenate([pep_target_feats, pep_struct_feats], axis=1)
    x = torch.tensor(np.concatenate([mhc_final, pep_final], axis=0), dtype=torch.float32, device=device)

    # Coords
    pos_mhc_ca = torch.tensor(mhc_template['coords_ca'][:limit_mhc], dtype=torch.float32, device=device)
    pos_pep_ca = torch.tensor(pep_template['coords_ca'][:limit_pep], dtype=torch.float32, device=device)
    pos_ca = torch.cat([pos_mhc_ca, pos_pep_ca], dim=0)

    pos_mhc_cb = torch.tensor(mhc_template['coords_cb'][:limit_mhc], dtype=torch.float32, device=device)
    pos_pep_cb = torch.tensor(pep_template['coords_cb'][:limit_pep], dtype=torch.float32, device=device)
    pos_cb = torch.cat([pos_mhc_cb, pos_pep_cb], dim=0)

    # Edges
    edge_index = radius_graph(pos_ca, r=10.0, loop=False)
    edge_attr = compute_edge_features(pos_ca, pos_cb, edge_index)

    pIC50 = -np.log10(ic50 * 1e-9 + 1e-12)
    batch_mask = torch.cat([torch.zeros(len(pos_mhc_ca)), torch.ones(len(pos_pep_ca))]).to(device)

    # CRITICAL: We explicitly pass pos=pos_ca
    data = Data(
        x=x.cpu(), 
        pos=pos_ca.cpu(), # <--- THIS IS THE FIX
        edge_index=edge_index.cpu(), 
        edge_attr=edge_attr.cpu(), 
        chain_mask=batch_mask.cpu().bool(),
        y=torch.tensor([pIC50]),
        identity=torch.tensor([0.0], dtype=torch.float32)
    )
    if mhc_template['seq']:
        sim = difflib.SequenceMatcher(None, mhc_seq, mhc_template['seq'], autojunk=False).ratio()
        data.identity = torch.tensor([sim], dtype=torch.float32)
    
    return data

def main():
    templates_cache = []
    print("Loading Templates...")
    for t_file in TEMPLATE_FILES:
        path = os.path.join(TEMPLATE_DIR, t_file)
        if os.path.exists(path):
            mhc_data, pep_data = load_template_data(path)
            if len(mhc_data['coords_ca']) > 0: templates_cache.append((mhc_data, pep_data))
    
    if not templates_cache: return

    df = pd.read_csv("../iedb_dataset.csv").dropna(subset=['mhc_sequence', 'peptide', 'binding_affinity_nM'])
    
    # NEW: Create logs list to save as CSV later
    similarity_logs = []

    print(f"Threading {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        row_log = {'original_index': idx}
        for t_idx, (mhc_template, pep_template) in enumerate(templates_cache):
            try:
                unified_graph = create_unified_interaction_graph(
                    row['mhc_sequence'].upper(), row['peptide'].upper(), 
                    mhc_template, pep_template, row['binding_affinity_nM']
                )
                torch.save(unified_graph, OUT_DIR / "mhc" / f"{idx}_T{t_idx}.pt")
                row_log[f'T{t_idx}_identity'] = float(unified_graph.identity)
            except Exception: 
                row_log[f'T{t_idx}_identity'] = 0.0
        similarity_logs.append(row_log)
        if len(similarity_logs) % 10000 == 0: pd.DataFrame(similarity_logs).to_csv(OUT_DIR / "sim_logs_partial.csv")
    
    pd.DataFrame(similarity_logs).to_csv(OUT_DIR / "sim_logs_final.csv")

if __name__ == "__main__":
    main()