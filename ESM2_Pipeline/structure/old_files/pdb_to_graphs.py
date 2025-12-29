"""
Convert PDB structures to graph representations for GNN processing (Enhanced)

Updates:
- Node Features: Atom Type (One-hot) + Residue Type (One-hot) + pLDDT
- Edges: Spatial proximity (<5Å)
- Edge Features: Raw distance + Inverse distance

Output: PyTorch Geometric Data objects saved as .pt files
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants & Mappings
# ============================================================================

# ... (Keep your existing AMINO_ACIDS and ATOM_TYPES) ...

# === NEW: CHEMICAL KNOWLEDGE BANK ===
# 1. Hydrophobicity (Kyte-Doolittle scale, normalized approx)
HYDROPHOBICITY = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

# 2. Charge (at physiological pH 7.4)
CHARGE = {
    'ARG': 1.0, 'LYS': 1.0, 'ASP': -1.0, 'GLU': -1.0, 
    'HIS': 0.1  # Histidine is partially protonated/neutral
}

# 3. Hydrogen Bonding (Simplified: [Donor, Acceptor])
# 1 = Yes, 0 = No
HBOND = {
    'ASN': [1,1], 'GLN': [1,1], 'SER': [1,1], 'THR': [1,1], 'TYR': [1,1], 
    'ASP': [0,1], 'GLU': [0,1], 'LYS': [1,0], 'ARG': [1,0], 'HIS': [1,1],
    'TRP': [1,0]
}

# Standard 20 Amino Acids (3-letter codes)
AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Common Atom Types in Proteins
ATOM_TYPES = ['C', 'N', 'O', 'S']
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

# ============================================================================
# PDB Parsing
# ============================================================================

def parse_pdb(pdb_file: Path, use_all_atoms: bool = False):
    """
    Parse PDB file and extract atom details including Residue Information.
    """
    atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            
            try:
                # Standard PDB Column format
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                
                # Coordinates
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                b_factor = float(line[60:66].strip())  # pLDDT
                
                # Element symbol (cols 76-78) or infer from name
                element = line[76:78].strip()
                if not element:
                    element = atom_name[0] # Fallback
                
                # Filter: CA only or all atoms
                if use_all_atoms or atom_name == 'CA':
                    atoms.append({
                        'element': element,
                        'res_name': res_name,
                        'pos': np.array([x, y, z], dtype=np.float32),
                        'plddt': b_factor
                    })
            except (ValueError, IndexError):
                continue
    
    return atoms

# ============================================================================
# Graph Construction
# ============================================================================

def atoms_to_graph(atoms: list, distance_threshold: float = 5.0) -> Data:
    """
    Convert atom list to PyTorch Geometric Data object with CHEMICAL features.
    """
    if len(atoms) == 0:
        return None
    
    # 1. Build Node Features
    # ---------------------------------------------------------
    node_features = []
    positions = []
    
    for atom in atoms:
        res = atom['res_name']
        
        # A. Atom Type One-Hot (Length: 5)
        atom_one_hot = np.zeros(len(ATOM_TYPES) + 1, dtype=np.float32)
        if atom['element'] in ATOM_TO_IDX:
            atom_one_hot[ATOM_TO_IDX[atom['element']]] = 1.0
        else:
            atom_one_hot[-1] = 1.0 
            
        # B. Residue Type One-Hot (Length: 21)
        res_one_hot = np.zeros(len(AMINO_ACIDS) + 1, dtype=np.float32)
        if res in AA_TO_IDX:
            res_one_hot[AA_TO_IDX[res]] = 1.0
        else:
            res_one_hot[-1] = 1.0
            
        # C. Physicochemical Properties (The Upgrade)
        # ------------------------------------------------
        # 1. Hydrophobicity (Normalized by division)
        hyd = [HYDROPHOBICITY.get(res, 0.0) / 4.5]
        
        # 2. Charge
        chg = [CHARGE.get(res, 0.0)]
        
        # 3. H-Bond Capability [Donor, Acceptor]
        hb = HBOND.get(res, [0, 0])
        
        # 4. pLDDT (Confidence)
        plddt_norm = [atom['plddt'] / 100.0]

        # Combine all features
        # Vector Size: 5 (Atom) + 21 (Res) + 1 (Hyd) + 1 (Chg) + 2 (HB) + 1 (pLDDT) = 31 dims
        feat = np.concatenate([atom_one_hot, res_one_hot, hyd, chg, hb, plddt_norm])
        
        node_features.append(feat)
        positions.append(atom['pos'])
    
    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    pos = torch.tensor(np.array(positions), dtype=torch.float32)
    
    # 2. Build Edges (Internal Distance Based)
    # ---------------------------------------------------------
    # Note: This only builds edges WITHIN the MHC or WITHIN the Peptide
    # because they are processed separately.
    
    positions_arr = np.array(positions)
    
    # Calculate distances
    dist_matrix = np.linalg.norm(positions_arr[:, None, :] - positions_arr[None, :, :], axis=-1)
    
    # Create mask (Threshold + Self-loop removal)
    mask = (dist_matrix < distance_threshold) & (dist_matrix > 1e-6)
    src_nodes, dst_nodes = np.where(mask)
    
    edges = []
    edge_attributes = []
    
    for src, dst in zip(src_nodes, dst_nodes):
        dist = dist_matrix[src, dst]
        
        edges.append([src, dst])
        
        # Edge Features: [Distance, 1/Distance]
        attr = [dist, 1.0 / (dist + 1e-6)]
        edge_attributes.append(attr)
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)
    
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

# ============================================================================
# Batch Processing
# ============================================================================

def convert_directory(pdb_dir: Path, output_dir: Path, use_all_atoms: bool = False, 
                      distance_threshold: float = 5.0):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDBs (adjust glob pattern if needed)
    pdb_files = list(pdb_dir.glob('*.pdb'))
    logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
    
    success = 0
    
    for i, pdb_file in enumerate(pdb_files, 1):
        if i % 100 == 0:
            logger.info(f"  Processed {i}/{len(pdb_files)}")
        
        output_file = output_dir / f"{pdb_file.stem}.pt"
        if output_file.exists():
            continue
        
        try:
            atoms = parse_pdb(pdb_file, use_all_atoms=use_all_atoms)
            if not atoms:
                continue
            
            graph = atoms_to_graph(atoms, distance_threshold=distance_threshold)
            
            if graph is not None:
                torch.save(graph, output_file)
                success += 1
                
        except Exception as e:
            logger.warning(f"Error processing {pdb_file.name}: {e}")
            continue
            
    logger.info(f"Conversion complete. Successfully converted {success} files.")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mhc-pdb-dir', type=str, required=True)
    parser.add_argument('--peptide-pdb-dir', type=str, required=True)
    parser.add_argument('--output-dir', default='data/graphs')
    parser.add_argument('--all-atoms', action='store_true', help='Use all atoms (default is CA only)')
    parser.add_argument('--distance-threshold', type=float, default=5.0)
    
    args = parser.parse_args()
    
    base_out = Path(args.output_dir)
    
    logger.info(f"Processing MHC PDBs from {args.mhc_pdb_dir}...")
    convert_directory(Path(args.mhc_pdb_dir), base_out / 'mhc', 
                     args.all_atoms, args.distance_threshold)
                     
    logger.info(f"Processing Peptide PDBs from {args.peptide_pdb_dir}...")
    convert_directory(Path(args.peptide_pdb_dir), base_out / 'peptide', 
                     args.all_atoms, args.distance_threshold)

if __name__ == '__main__':
    main()