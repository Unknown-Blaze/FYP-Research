"""
Simple PDB to Graph converter - minimal dependencies
Converts PDB files to PyTorch tensors for graph neural networks
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not installed")

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric not installed - graphs will be simple numpy arrays")


class SimplePDBParser:
    """Parse PDB files without external dependencies"""
    
    def __init__(self):
        self.aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
            'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
            'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20
        }
    
    def parse_pdb(self, pdb_file: str) -> Optional[Dict]:
        """
        Parse PDB file and extract CA atoms
        
        Returns:
            {
                'coords': np.array (N, 3),
                'residues': list of residue names,
                'chains': list of chain IDs,
                'bfactors': np.array (N,)
            }
        """
        try:
            coords = []
            residues = []
            chains = []
            bfactors = []
            
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        # PDB format parsing
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        chain = line[21]
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        bfactor = float(line[60:66])
                        
                        # Only keep CA atoms
                        if atom_name == 'CA':
                            coords.append([x, y, z])
                            residues.append(res_name)
                            chains.append(chain)
                            bfactors.append(bfactor)
            
            if len(coords) == 0:
                logger.warning(f"No CA atoms in {pdb_file}")
                return None
            
            return {
                'coords': np.array(coords, dtype=np.float32),
                'residues': residues,
                'chains': chains,
                'bfactors': np.array(bfactors, dtype=np.float32)
            }
        
        except Exception as e:
            logger.error(f"Error parsing {pdb_file}: {e}")
            return None
    
    def three_to_one(self, res_name: str) -> str:
        """Convert 3-letter to 1-letter amino acid code"""
        mapping = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        return mapping.get(res_name.upper(), 'X')


class PDBToGraph:
    """Convert PDB to graph representation"""
    
    def __init__(self, contact_distance: float = 10.0):
        self.parser = SimplePDBParser()
        self.contact_distance = contact_distance
    
    def convert(self, pdb_file: str) -> Optional[Dict]:
        """
        Convert PDB to graph
        
        Returns dict with:
        - node_features: amino acid one-hot
        - positions: coordinates
        - edges: connectivity
        - edge_dist: distances
        """
        pdb_data = self.parser.parse_pdb(pdb_file)
        
        if pdb_data is None:
            return None
        
        coords = pdb_data['coords']
        residues = pdb_data['residues']
        n_nodes = len(coords)
        
        # Node features: one-hot amino acid encoding
        node_features = np.zeros((n_nodes, 21), dtype=np.float32)
        
        for i, res in enumerate(residues):
            aa_code = self.parser.three_to_one(res)
            aa_idx = self.parser.aa_to_idx.get(aa_code, 20)
            node_features[i, aa_idx] = 1.0
        
        # Build edges based on distance
        edges = []
        edge_dist = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = np.linalg.norm(coords[i] - coords[j])
                
                if dist < self.contact_distance:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    edge_dist.append(dist)
                    edge_dist.append(dist)
        
        if len(edges) == 0:
            # No contacts - add self-loops
            edges = [[i, i] for i in range(n_nodes)]
            edge_dist = [0.0] * n_nodes
        
        graph_dict = {
            'node_features': node_features,  # (N, 21)
            'positions': coords,              # (N, 3)
            'edges': np.array(edges, dtype=np.int64),  # (E, 2)
            'edge_dist': np.array(edge_dist, dtype=np.float32),  # (E,)
            'num_nodes': n_nodes,
            'num_edges': len(edges),
            'residues': residues,
            'chains': pdb_data['chains']
        }
        
        return graph_dict


def build_graphs_batch(
    pdb_dir: str = "./structure/data/structures",
    output_dir: str = "./structure/data/graphs",
    mapping_file: str = "./structure/data/structures/mapping.json"
) -> Dict[int, str]:
    """
    Convert all PDB files to graph representations
    
    Outputs:
    - .npy files containing graph data (numpy)
    - .pt files containing PyTorch tensors (if torch available)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load mapping
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Mapping file not found: {mapping_file}")
        logger.error("Run run_alphafold.py first")
        return {}
    
    pdb_files = mapping_data.get('success', {})
    
    if not pdb_files:
        logger.error("No PDB files in mapping")
        return {}
    
    logger.info(f"Converting {len(pdb_files)} PDB files to graphs")
    
    converter = PDBToGraph(contact_distance=10.0)
    graph_files = {}
    failed = []
    
    for idx_str, pdb_path in pdb_files.items():
        idx = int(idx_str)
        
        if idx % 10 == 0:
            logger.info(f"Progress: {idx}/{len(pdb_files)}")
        
        try:
            # Convert
            graph = converter.convert(pdb_path)
            
            if graph is None:
                failed.append(idx)
                continue
            
            # Save as numpy
            graph_file = os.path.join(output_dir, f"graph_{idx:06d}.npz")
            np.savez_compressed(
                graph_file,
                node_features=graph['node_features'],
                positions=graph['positions'],
                edges=graph['edges'],
                edge_dist=graph['edge_dist'],
                num_nodes=graph['num_nodes'],
                residues=graph['residues'],
                chains=graph['chains']
            )
            
            # Also save as PyTorch if available
            if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
                torch_graph = Data(
                    x=torch.tensor(graph['node_features'], dtype=torch.float32),
                    pos=torch.tensor(graph['positions'], dtype=torch.float32),
                    edge_index=torch.tensor(graph['edges'].T, dtype=torch.long),
                    edge_attr=torch.tensor(graph['edge_dist'].reshape(-1, 1), dtype=torch.float32)
                )
                torch_file = os.path.join(output_dir, f"graph_{idx:06d}.pt")
                torch.save(torch_graph, torch_file)
            
            graph_files[idx] = graph_file
        
        except Exception as e:
            logger.warning(f"Failed [{idx}]: {e}")
            failed.append(idx)
    
    # Save mapping
    output_mapping = {
        "success": {str(k): v for k, v in graph_files.items()},
        "failed": failed,
        "total": len(pdb_files),
        "format": "numpy (.npz) and PyTorch (.pt)" if TORCH_AVAILABLE else "numpy (.npz)"
    }
    
    output_mapping_file = os.path.join(output_dir, "mapping.json")
    with open(output_mapping_file, 'w') as f:
        json.dump(output_mapping, f, indent=2)
    
    logger.info(f"\n✓ Converted {len(graph_files)}/{len(pdb_files)} graphs")
    logger.info(f"✓ Output: {output_dir}")
    
    return graph_files


if __name__ == "__main__":
    build_graphs_batch()