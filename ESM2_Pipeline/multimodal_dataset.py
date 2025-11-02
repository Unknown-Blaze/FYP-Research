"""
Multimodal Dataset: Combines Sequence (ESM2) + Structure (Graphs)
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric not available - using numpy format")


class MultimodalMHCDataset(Dataset):
    """
    Combines sequence (ESM2) and structure (graphs) for MHC-peptide binding prediction
    
    Usage:
        dataset = MultimodalMHCDataset(
            csv_file="iedb_dataset.csv",
            tokenizer=tokenizer,
            graph_dir="./structure/data/graphs",
            graph_mapping_file="./structure/data/graphs/mapping.json"
        )
    """
    
    def __init__(
        self,
        csv_file: str,
        tokenizer,
        graph_dir: str = "./structure/data/graphs",
        graph_mapping_file: str = "./structure/data/graphs/mapping.json",
        max_length: int = 512,
        use_torch_graphs: bool = True
    ):
        """
        Args:
            csv_file: Path to IEDB dataset CSV
            tokenizer: ESM2 tokenizer
            graph_dir: Directory with graph files
            graph_mapping_file: Mapping of indices to graph files
            max_length: Max sequence length
            use_torch_graphs: Use .pt (PyTorch) or .npz (numpy) graphs
        """
        self.tokenizer = tokenizer
        self.graph_dir = graph_dir
        self.max_length = max_length
        self.use_torch_graphs = use_torch_graphs and TORCH_GEOMETRIC_AVAILABLE
        
        # Load dataset
        logger.info(f"Loading {csv_file}")
        self.df = pd.read_csv(csv_file, on_bad_lines='skip')
        
        # Load graph mapping
        logger.info(f"Loading graph mapping from {graph_mapping_file}")
        with open(graph_mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        self.graph_files = mapping_data.get('success', {})
        logger.info(f"Found graphs for {len(self.graph_files)} samples")
        
        # Filter to samples with both sequence and structure data
        valid_indices = [int(idx) for idx in self.graph_files.keys()]
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        logger.info(f"Dataset size (with structures): {len(self.df)}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'mhc_input_ids': tensor (seq_len,),
                'mhc_attention_mask': tensor (seq_len,),
                'pep_input_ids': tensor (seq_len,),
                'pep_attention_mask': tensor (seq_len,),
                'graph': Data or dict with graph info,
                'label': tensor (scalar)
            }
        """
        row = self.df.iloc[idx]
        
        # Get sequences
        mhc_seq = str(row['mhc_sequence']).strip()
        pep_seq = str(row['peptide']).strip()
        label = float(row['binding_affinity_nM'])
        
        # Tokenize MHC
        mhc_tokens = self.tokenizer(
            mhc_seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize peptide
        pep_tokens = self.tokenizer(
            pep_seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Load graph
        graph = self._load_graph(idx)
        
        return {
            'mhc_input_ids': mhc_tokens['input_ids'].squeeze(0),
            'mhc_attention_mask': mhc_tokens['attention_mask'].squeeze(0),
            'pep_input_ids': pep_tokens['input_ids'].squeeze(0),
            'pep_attention_mask': pep_tokens['attention_mask'].squeeze(0),
            'graph': graph,
            'label': torch.tensor(label, dtype=torch.float32)
        }
    
    def _load_graph(self, idx: int):
        """Load graph for sample idx"""
        graph_file = self.graph_files[str(idx)]
        
        if self.use_torch_graphs:
            # Try PyTorch format first
            torch_file = graph_file.replace('.npz', '.pt')
            try:
                return torch.load(torch_file)
            except FileNotFoundError:
                logger.warning(f"PyTorch graph not found {torch_file}, trying numpy")
                return self._load_numpy_graph(graph_file)
        else:
            # Use numpy format
            return self._load_numpy_graph(graph_file)
    
    def _load_numpy_graph(self, npz_file: str) -> Dict:
        """Load numpy graph and convert to dict"""
        data = np.load(npz_file, allow_pickle=True)
        
        graph_dict = {
            'node_features': torch.tensor(data['node_features'], dtype=torch.float32),
            'positions': torch.tensor(data['positions'], dtype=torch.float32),
            'edges': torch.tensor(data['edges'], dtype=torch.long),
            'edge_dist': torch.tensor(data['edge_dist'], dtype=torch.float32),
            'num_nodes': int(data['num_nodes'])
        }
        
        return graph_dict


def collate_multimodal(batch_list):
    """Custom collate function for multimodal batch"""
    
    # Pad sequences
    mhc_input_ids = torch.stack([item['mhc_input_ids'] for item in batch_list])
    mhc_attn_mask = torch.stack([item['mhc_attention_mask'] for item in batch_list])
    pep_input_ids = torch.stack([item['pep_input_ids'] for item in batch_list])
    pep_attn_mask = torch.stack([item['pep_attention_mask'] for item in batch_list])
    labels = torch.stack([item['label'] for item in batch_list])
    
    # Handle graphs
    graphs = [item['graph'] for item in batch_list]
    
    return {
        'mhc_batch': {
            'input_ids': mhc_input_ids,
            'attention_mask': mhc_attn_mask
        },
        'pep_batch': {
            'input_ids': pep_input_ids,
            'attention_mask': pep_attn_mask
        },
        'graphs': graphs,
        'labels': labels
    }


if __name__ == "__main__":
    # Test
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    dataset = MultimodalMHCDataset(
        csv_file="iedb_dataset.csv",
        tokenizer=tokenizer,
        graph_dir="./structure/data/graphs",
        graph_mapping_file="./structure/data/graphs/mapping.json"
    )
    
    logger.info(f"Dataset ready: {len(dataset)} samples")
    
    # Test sample
    sample = dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"MHC shape: {sample['mhc_input_ids'].shape}")
    logger.info(f"Graph nodes: {sample['graph'].num_nodes if hasattr(sample['graph'], 'num_nodes') else 'dict format'}")
