"""
Convert PDB structures to graph representations for GNN processing

Creates molecular graphs where:
- Nodes = atoms (CA only for simplicity, or all atoms)
- Edges = spatial proximity (<5Å) or bond connections
- Features = atom type, pLDDT confidence, position

Output: PyTorch Geometric Data objects saved as .pt files
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PDB Parsing
# ============================================================================

def parse_pdb(pdb_file: Path, use_all_atoms: bool = False):
    """
    Parse PDB file and extract atom coordinates, types, and B-factors (pLDDT).
    
    Args:
        pdb_file: Path to PDB file
        use_all_atoms: If True, use all atoms. If False, use CA only.
    
    Returns:
        atoms: list of (atom_type, x, y, z, b_factor)
    """
    atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            
            try:
                atom_name = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                b_factor = float(line[60:66].strip())  # pLDDT
                
                # Filter: CA only or all atoms
                if use_all_atoms or atom_name == 'CA':
                    atom_type = line[76:78].strip() if len(line) > 77 else 'C'
                    atoms.append({
                        'type': atom_type or 'C',
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
    Convert atom list to PyTorch Geometric Data object.
    
    Creates edges between atoms within distance_threshold.
    
    Args:
        atoms: List of atom dicts with 'type', 'pos', 'plddt'
        distance_threshold: Max distance (Å) for edge creation
    
    Returns:
        PyTorch Geometric Data object
    """
    if len(atoms) == 0:
        return None
    
    # Node features: one-hot atom type + pLDDT
    atom_types = sorted(set(a['type'] for a in atoms))
    type_to_idx = {t: i for i, t in enumerate(atom_types)}
    
    node_features = []
    positions = []
    plddt_scores = []
    
    for atom in atoms:
        # One-hot encode atom type
        one_hot = np.zeros(len(atom_types), dtype=np.float32)
        one_hot[type_to_idx[atom['type']]] = 1
        
        node_features.append(one_hot)
        positions.append(atom['pos'])
        plddt_scores.append(atom['plddt'])
    
    # Stack features
    x = np.column_stack([node_features, plddt_scores])  # (N, atom_types + 1)
    x = torch.tensor(x, dtype=torch.float32)
    pos = torch.tensor(np.array(positions), dtype=torch.float32)
    
    # Create edges: distance-based
    positions_arr = np.array(positions)
    n_atoms = len(atoms)
    edges = []
    edge_weights = []
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions_arr[i] - positions_arr[j])
            if dist < distance_threshold:
                edges.append([i, j])
                edges.append([j, i])  # Undirected
                edge_weights.append(1.0 / (dist + 1e-6))  # Weight by inverse distance
                edge_weights.append(1.0 / (dist + 1e-6))
    
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    
    # Create PyG Data object
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    return data

# ============================================================================
# Batch Processing
# ============================================================================

def convert_directory(pdb_dir: Path, output_dir: Path, use_all_atoms: bool = False, 
                      distance_threshold: float = 5.0):
    """
    Convert all PDB files in a directory to graph files.
    
    Args:
        pdb_dir: Input directory with PDB files
        output_dir: Output directory for .pt graph files
        use_all_atoms: Use all atoms or CA only
        distance_threshold: Distance threshold for edges
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdb_files = list(pdb_dir.glob('*rank_001*.pdb'))
    logger.info(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
    
    success = 0
    failed = 0
    skipped = 0
    
    for i, pdb_file in enumerate(pdb_files, 1):
        if i % 1000 == 0:
            logger.info(f"  Processed {i}/{len(pdb_files)}")
        
        # Check if already converted
        output_file = output_dir / f"{pdb_file.stem}.pt"
        if output_file.exists():
            skipped += 1
            continue
        
        try:
            atoms = parse_pdb(pdb_file, use_all_atoms=use_all_atoms)
            
            if not atoms:
                logger.warning(f"No atoms found in {pdb_file.name}")
                failed += 1
                continue
            
            graph = atoms_to_graph(atoms, distance_threshold=distance_threshold)
            
            if graph is None:
                failed += 1
                continue
            
            torch.save(graph, output_file)
            success += 1
            
        except Exception as e:
            logger.warning(f"Error processing {pdb_file.name}: {e}")
            failed += 1
    
    logger.info(f"✓ Conversion complete!")
    logger.info(f"  Success: {success}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped (already converted): {skipped}")
    logger.info(f"  Output directory: {output_dir}")

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert PDB files to graph representations for GNN')
    parser.add_argument('--mhc-pdb-dir', default='data/generated_pdbs/mhc',
                       help='Path to MHC PDB directory')
    parser.add_argument('--peptide-pdb-dir', default='data/generated_pdbs/peptide',
                       help='Path to peptide PDB directory')
    parser.add_argument('--output-dir', default='data/graphs',
                       help='Output directory for graph files')
    parser.add_argument('--all-atoms', action='store_true',
                       help='Use all atoms instead of CA only')
    parser.add_argument('--distance-threshold', type=float, default=5.0,
                       help='Distance threshold for edges (Å)')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("CONVERTING PDB FILES TO GRAPHS FOR GNN")
    logger.info("=" * 70)
    logger.info("")
    
    base_out = Path(args.output_dir)
    mhc_out = base_out / 'mhc'
    peptide_out = base_out / 'peptide'
    
    # Convert MHC structures
    logger.info("Converting MHC structures...")
    logger.info("-" * 70)
    convert_directory(
        Path(args.mhc_pdb_dir),
        mhc_out,
        use_all_atoms=args.all_atoms,
        distance_threshold=args.distance_threshold
    )
    
    logger.info("")
    
    # Convert peptide structures
    logger.info("Converting peptide structures...")
    logger.info("-" * 70)
    convert_directory(
        Path(args.peptide_pdb_dir),
        peptide_out,
        use_all_atoms=args.all_atoms,
        distance_threshold=args.distance_threshold
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ ALL CONVERSIONS COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Graph files ready for GNN training:")
    logger.info(f"  MHC graphs: {mhc_out}")
    logger.info(f"  Peptide graphs: {peptide_out}")
    logger.info("")

if __name__ == '__main__':
    main()
