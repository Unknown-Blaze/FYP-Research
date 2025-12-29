"""
Convert Integrated PDBs (Chain A=MHC, Chain B=Pep) to separate Graphs
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Mappings (Keep your existing chemical dictionaries)
# ============================================================================
HYDROPHOBICITY = {'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2}
CHARGE = {'ARG': 1.0, 'LYS': 1.0, 'ASP': -1.0, 'GLU': -1.0, 'HIS': 0.1}
HBOND = {'ASN': [1,1], 'GLN': [1,1], 'SER': [1,1], 'THR': [1,1], 'TYR': [1,1], 'ASP': [0,1], 'GLU': [0,1], 'LYS': [1,0], 'ARG': [1,0], 'HIS': [1,1], 'TRP': [1,0]}
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ATOM_TYPES = ['C', 'N', 'O', 'S']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

# ============================================================================
# Core Logic
# ============================================================================

def get_atom_features(atom_name, res_name, b_factor):
    """Build the 31-dim feature vector"""
    # 1. Atom Type (5)
    atom_one_hot = np.zeros(len(ATOM_TYPES) + 1, dtype=np.float32)
    element = atom_name[0]
    if element in ATOM_TO_IDX: atom_one_hot[ATOM_TO_IDX[element]] = 1.0
    else: atom_one_hot[-1] = 1.0

    # 2. Residue Type (21)
    res_one_hot = np.zeros(len(AMINO_ACIDS) + 1, dtype=np.float32)
    if res_name in AA_TO_IDX: res_one_hot[AA_TO_IDX[res_name]] = 1.0
    else: res_one_hot[-1] = 1.0

    # 3. Chemical Props (5)
    hyd = [HYDROPHOBICITY.get(res_name, 0.0) / 4.5]
    chg = [CHARGE.get(res_name, 0.0)]
    hb = HBOND.get(res_name, [0, 0])
    plddt = [b_factor / 100.0]

    return np.concatenate([atom_one_hot, res_one_hot, hyd, chg, hb, plddt])

def process_complex_pdb(pdb_path):
    """
    Reads ONE PDB file containing both chains.
    Returns: mhc_graph (Data), pep_graph (Data)
    """
    mhc_atoms = []
    pep_atoms = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                chain_id = line[21]
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                
                # Coordinates
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                # pLDDT is in B-factor column
                plddt = float(line[60:66])
                
                # Filter for CA only (to keep graphs light) or All Atoms
                # Using CA only for speed, but you can enable all atoms if needed
                if atom_name == 'CA': 
                    feature = get_atom_features(atom_name, res_name, plddt)
                    pos = [x, y, z]
                    
                    atom_data = {'x': feature, 'pos': pos}
                    
                    if chain_id == 'A':
                        mhc_atoms.append(atom_data)
                    elif chain_id == 'B':
                        pep_atoms.append(atom_data)

    def build_graph(atom_list):
        if not atom_list: return None
        x = torch.tensor(np.array([a['x'] for a in atom_list]), dtype=torch.float32)
        pos = torch.tensor(np.array([a['pos'] for a in atom_list]), dtype=torch.float32)
        
        # Simple sequential edges (i -> i+1)
        # The GNN RBF layers will handle the 3D distance edges!
        num_nodes = len(atom_list)
        edge_index = torch.stack([
            torch.arange(num_nodes - 1),
            torch.arange(1, num_nodes)
        ], dim=0).long()
        
        # Dummy edge attr (GNN will generate RBF)
        edge_attr = torch.zeros((edge_index.size(1), 16), dtype=torch.float32)
        
        return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    return build_graph(mhc_atoms), build_graph(pep_atoms)

def main():
    # CONFIG
    PDB_DIR = Path("data/generated_pdbs/complexes")
    OUT_DIR = Path("data/graphs")
    
    (OUT_DIR / "mhc").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "peptide").mkdir(parents=True, exist_ok=True)
    
    files = list(PDB_DIR.glob("complex_*.pdb"))
    logger.info(f"Found {len(files)} complex PDBs")
    
    success = 0
    for p in tqdm(files):
        try:
            # ID is "complex_12345.pdb" -> "12345"
            complex_id = p.stem.split('_')[-1]
            
            mhc_g, pep_g = process_complex_pdb(p)
            
            if mhc_g and pep_g:
                torch.save(mhc_g, OUT_DIR / "mhc" / f"{complex_id}_rank_001.pt")
                torch.save(pep_g, OUT_DIR / "peptide" / f"{complex_id}_rank_001.pt")
                success += 1
        except Exception as e:
            # logger.error(f"Error {p}: {e}")
            pass
            
    logger.info(f"Successfully converted {success} complexes.")

if __name__ == '__main__':
    main()