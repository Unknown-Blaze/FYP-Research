"""
Graph Neural Network for MHC-Peptide Binding Affinity Prediction

Architecture: Graph Attention Network (GAT)
- Reads graph representations of MHC and peptide structures
- Aggregates structural information via attention mechanism
- Predicts IC50 binding affinity

Installation:
  pip install torch-geometric pytorch-scatter pytorch-sparse
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from datetime import datetime

# Setup logging
log_file = f"gnn_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'dataset_path': '../iedb_dataset.csv',
    'graphs_dir_mhc': 'data/graphs/mhc',
    'graphs_dir_peptide': 'data/graphs/peptide',
    'mapping_file': 'row_to_pdb_map.json',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'hidden_dims': [64, 64],
    'num_heads': 4,
    'dropout': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# Dataset
# ============================================================================

class GraphBindingDataset(Dataset):
    def __init__(self, df, mapping, graphs_dir_mhc, graphs_dir_peptide):
        self.df = df.reset_index(drop=True)
        self.mapping = mapping
        self.graphs_dir_mhc = Path(graphs_dir_mhc)
        self.graphs_dir_peptide = Path(graphs_dir_peptide)
        
        logger.info("Pre-caching graph files...")
        self.graph_cache = {}
        
        for idx in range(len(self.df)):
            key = str(idx)
            if key in self.mapping:
                mhc_id = self.mapping[key]['mhc_pdb_id']
                pep_id = self.mapping[key]['peptide_pdb_id']
                
                if mhc_id not in self.graph_cache:
                    mhc_graph_file = self.graphs_dir_mhc / f"{mhc_id}_*rank_001*.pt"
                    mhc_files = list(self.graphs_dir_mhc.glob(f"{mhc_id}*rank_001*.pt"))
                    if mhc_files:
                        self.graph_cache[mhc_id] = torch.load(mhc_files[0])
                
                if pep_id not in self.graph_cache:
                    pep_files = list(self.graphs_dir_peptide.glob(f"{pep_id}*rank_001*.pt"))
                    if pep_files:
                        self.graph_cache[pep_id] = torch.load(pep_files[0])
        
        logger.info(f"Cached {len(self.graph_cache)} unique graphs")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = str(idx)
        
        # Get graphs
        mhc_graph = None
        pep_graph = None
        
        if key in self.mapping:
            mhc_id = self.mapping[key]['mhc_pdb_id']
            pep_id = self.mapping[key]['peptide_pdb_id']
            
            mhc_graph = self.graph_cache.get(mhc_id)
            pep_graph = self.graph_cache.get(pep_id)
        
        # If graphs missing, create dummy graph
        if mhc_graph is None:
            mhc_graph = Data(
                x=torch.zeros((1, 2), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1), dtype=torch.float32)
            )
        
        if pep_graph is None:
            pep_graph = Data(
                x=torch.zeros((1, 2), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 1), dtype=torch.float32)
            )
        
        ic50 = float(row['IC50'])
        
        return mhc_graph, pep_graph, ic50


# ============================================================================
# GNN Model
# ============================================================================

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dims: list = None, 
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        self.dropout = dropout
        
        # Graph attention layers
        dims = [input_dim] + hidden_dims
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.gat_layers.append(
                GATConv(dims[i], dims[i+1] // num_heads, heads=num_heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(dims[i+1]))
        
        # Global feature dimension: (2 * last_hidden) for mean and max pooling
        self.global_dim = dims[-1]
        
        # Binding prediction head
        combined_dim = 2 * self.global_dim * 2  # MHC + peptide, each with mean+max pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward_graph(self, x, edge_index, edge_attr, batch):
        """Process single graph (MHC or peptide)"""
        for gat, norm in zip(self.gat_layers, self.norms):
            x = gat(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if norm is not None:
                x = norm(x)
        
        # Global pooling: mean + max
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        return torch.cat([mean_pool, max_pool], dim=1)
    
    def forward(self, mhc_graph, pep_graph):
        """
        Args:
            mhc_graph: PyG Data object for MHC
            pep_graph: PyG Data object for peptide
        
        Returns:
            IC50 prediction
        """
        # Process MHC graph
        mhc_features = self.forward_graph(
            mhc_graph.x, 
            mhc_graph.edge_index,
            mhc_graph.edge_attr,
            mhc_graph.batch if hasattr(mhc_graph, 'batch') else None
        )
        
        # Process peptide graph
        pep_features = self.forward_graph(
            pep_graph.x,
            pep_graph.edge_index,
            pep_graph.edge_attr,
            pep_graph.batch if hasattr(pep_graph, 'batch') else None
        )
        
        # Combine and predict
        combined = torch.cat([mhc_features, pep_features], dim=1)
        ic50 = self.fc_layers(combined)
        
        return ic50.squeeze()

# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for mhc_graphs, pep_graphs, targets in loader:
        mhc_graphs = [g.to(device) for g in mhc_graphs]
        pep_graphs = [g.to(device) for g in pep_graphs]
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = []
        for mhc, pep in zip(mhc_graphs, pep_graphs):
            out = model(mhc, pep)
            outputs.append(out)
        
        outputs = torch.stack(outputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for mhc_graphs, pep_graphs, targets in loader:
            mhc_graphs = [g.to(device) for g in mhc_graphs]
            pep_graphs = [g.to(device) for g in pep_graphs]
            
            outputs = []
            for mhc, pep in zip(mhc_graphs, pep_graphs):
                out = model(mhc, pep)
                outputs.append(out)
            
            outputs = torch.stack(outputs).cpu().numpy()
            predictions.extend(outputs)
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)
    
    return rmse, r2, predictions, actuals

# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("=" * 70)
    logger.info("GNN-BASED MHC-PEPTIDE BINDING AFFINITY PREDICTION")
    logger.info("=" * 70)
    logger.info("")
    
    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_csv(CONFIG['dataset_path'])
    logger.info(f"  Total samples: {len(df)}")
    
    # Load mapping
    logger.info("Loading PDB mapping...")
    with open(CONFIG['mapping_file'], 'r') as f:
        mapping = json.load(f)
    logger.info(f"  Mapped rows: {len(mapping)}")
    
    # Check graph availability
    graphs_mhc = list(Path(CONFIG['graphs_dir_mhc']).glob('*.pt'))
    graphs_pep = list(Path(CONFIG['graphs_dir_peptide']).glob('*.pt'))
    
    logger.info(f"  Available MHC graphs: {len(graphs_mhc)}")
    logger.info(f"  Available peptide graphs: {len(graphs_pep)}")
    logger.info("")
    
    if not graphs_mhc or not graphs_pep:
        logger.error("❌ Graph files not found!")
        logger.error("   Run: python pdb_to_graphs.py")
        return
    
    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples:  {len(test_df)}")
    logger.info("")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = GraphBindingDataset(train_df, mapping, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'])
    test_dataset = GraphBindingDataset(test_df, mapping, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Model
    device = torch.device(CONFIG['device'])
    logger.info(f"Using device: {device}")
    
    model = GraphAttentionNetwork(
        input_dim=2,
        hidden_dims=CONFIG['hidden_dims'],
        num_heads=CONFIG['num_heads'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    
    logger.info("")
    logger.info("Model architecture:")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("")
    logger.info("Starting training...")
    logger.info("-" * 70)
    
    best_rmse = float('inf')
    best_r2 = float('-inf')
    test_predictions_best = None
    test_actuals_best = None
    
    for epoch in range(CONFIG['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_rmse, test_r2, preds, actuals = evaluate(model, test_loader, device)
        
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_r2 = test_r2
            test_predictions_best = preds
            test_actuals_best = actuals
            torch.save(model.state_dict(), 'best_gnn_model.pth')
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | "
                       f"Test RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")
    
    logger.info("-" * 70)
    logger.info(f"✓ Training complete!")
    logger.info(f"  Best Test RMSE: {best_rmse:.4f}")
    logger.info(f"  Best Test R²: {best_r2:.4f}")
    logger.info(f"  MAE: {mean_absolute_error(test_actuals_best, test_predictions_best):.4f}")
    logger.info(f"  Model saved: best_gnn_model.pth")
    logger.info(f"  Log file: {log_file}")
    logger.info("")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Graph Attention Network (GAT)',
        'test_rmse': float(best_rmse),
        'test_r2': float(best_r2),
        'test_mae': float(mean_absolute_error(test_actuals_best, test_predictions_best)),
        'n_samples': len(df),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'model_params': sum(p.numel() for p in model.parameters()),
        'hidden_dims': CONFIG['hidden_dims'],
        'num_heads': CONFIG['num_heads'],
        'epochs': CONFIG['num_epochs'],
        'batch_size': CONFIG['batch_size'],
        'learning_rate': CONFIG['learning_rate'],
        'device': str(device)
    }
    
    results_file = f"gnn_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_file}")

if __name__ == '__main__':
    main()
