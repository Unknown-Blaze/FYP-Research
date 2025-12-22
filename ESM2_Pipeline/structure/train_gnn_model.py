"""
Graph Neural Network for MHC-Peptide Binding Affinity Prediction
(Two-Tower Architecture - Optimized for Separate PDBs)
"""
import warnings
# Filter specific warnings to keep logs clean
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# Correct Imports for PyG 2.0+
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data
from torch.utils.data import Dataset
from pathlib import Path
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
from datetime import datetime

# Setup logging
log_file = f"./logs/gnn_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
    'input_dim': 31,  # Matches your chemical feature size
    'dataset_path': '../iedb_dataset.csv',
    'graphs_dir_mhc': 'data/graphs/mhc',
    'graphs_dir_peptide': 'data/graphs/peptide',
    'mapping_file': 'row_to_pdb_map.json',
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.0001, # Optimized for Two-Tower
    'hidden_dims': [128, 128],
    'num_heads': 4,
    'dropout': 0.3,
    'weight_decay': 1e-4,
    'seed': 42,
    'grad_clip_norm': 1.0,
    'scheduler_factor': 0.5,
    'scheduler_patience': 3,
    'scheduler_min_lr': 1e-6,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# Dataset
# ============================================================================

import torch
from torch_geometric.data import Data

# Helper to create Radial Basis Function (RBF) features for distances
def rbf_expansion(distances, num_centers=16, cutoff=10.0):
    centers = torch.linspace(0, cutoff, num_centers, device=distances.device)
    width = centers[1] - centers[0]
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / width)

class GraphBindingDataset(Dataset):
    def __init__(self, df, mapping, graphs_dir_mhc, graphs_dir_peptide, cache_graphs: bool = False, label_col: str = 'IC50'):
        self.df = df.copy()
        self.mapping = mapping
        self.graphs_dir_mhc = Path(graphs_dir_mhc)
        self.graphs_dir_peptide = Path(graphs_dir_peptide)
        self.cache_graphs = cache_graphs 
        self.label_col = label_col
        self.graph_cache = {}

        if self.cache_graphs:
            logger.info("Pre-caching enabled (Warning: High RAM usage)...")

    def __len__(self):
        return len(self.df)

    def _add_distance_features(self, graph):
        # 1. Calculate Distances for existing edges
        row, col = graph.edge_index
        coord_diff = graph.pos[row] - graph.pos[col]
        radial_dist = torch.norm(coord_diff, dim=-1)
        
        # 2. Encode distances using RBF (Soft one-hot encoding of distance)
        edge_attr_dist = rbf_expansion(radial_dist, num_centers=16)
        
        # 3. Replace or Concatenate with existing edge_attr
        # Assuming existing edge_attr encodes bond type (e.g., size 2-4)
        # If your existing edge_attr is empty/zeros, just use dist
        graph.edge_attr = edge_attr_dist 
        return graph

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        key = str(row.name) 
        
        mhc_id, pep_id = None, None
        
        if key in self.mapping:
            mhc_id = self.mapping[key]['mhc_pdb_id']
            pep_id = self.mapping[key]['peptide_pdb_id']
        
        # --- LOAD GRAPHS ---
        mhc_graph = None
        if mhc_id:
            mhc_graph = self.graph_cache.get(mhc_id)
            if mhc_graph is None:
                files = list(self.graphs_dir_mhc.glob(f"{mhc_id}*rank_001*.pt"))
                if files: mhc_graph = torch.load(files[0])
        
        pep_graph = None
        if pep_id:
            pep_graph = self.graph_cache.get(pep_id)
            if pep_graph is None:
                files = list(self.graphs_dir_peptide.glob(f"{pep_id}*rank_001*.pt"))
                if files: pep_graph = torch.load(files[0])

        # --- DUMMY FALLBACK ---
        if mhc_graph is None:
            mhc_graph = Data(
                x=torch.randn((1, CONFIG['input_dim']), dtype=torch.float32) * 0.01, 
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                pos=torch.zeros((1, 3), dtype=torch.float32)
            )
        if pep_graph is None:
            pep_graph = Data(
                x=torch.randn((1, CONFIG['input_dim']), dtype=torch.float32) * 0.01, 
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, 2), dtype=torch.float32),
                pos=torch.zeros((1, 3), dtype=torch.float32)
            )

        # --- SAFETY CHECK: Features ---
        if not hasattr(mhc_graph, 'pos') or mhc_graph.pos is None:
            mhc_graph.pos = torch.zeros((mhc_graph.num_nodes, 3), dtype=torch.float32)
            
        if not hasattr(pep_graph, 'pos') or pep_graph.pos is None:
            pep_graph.pos = torch.zeros((pep_graph.num_nodes, 3), dtype=torch.float32)

        if not hasattr(mhc_graph, 'edge_attr') or mhc_graph.edge_attr is None:
            num_edges = mhc_graph.edge_index.shape[1] if mhc_graph.edge_index is not None else 0
            mhc_graph.edge_attr = torch.zeros((num_edges, 2), dtype=torch.float32)

        if not hasattr(pep_graph, 'edge_attr') or pep_graph.edge_attr is None:
            num_edges = pep_graph.edge_index.shape[1] if pep_graph.edge_index is not None else 0
            pep_graph.edge_attr = torch.zeros((num_edges, 2), dtype=torch.float32)

        mhc_graph = self._add_distance_features(mhc_graph)
        pep_graph = self._add_distance_features(pep_graph)
        
        # --- TARGET CALCULATION (pIC50) ---
        try:
            ic50_val = float(row[self.label_col])
            ic50_clamped = min(ic50_val, 50000.0)
            molar = ic50_clamped * 1e-9
            target = -np.log10(molar + 1e-12)
        except Exception:
            target = 0.0
        
        return mhc_graph, pep_graph, float(target)

# ============================================================================
# SOTA GNN Model (Cross-Attention / Docking Architecture)
# ============================================================================

from torch_geometric.utils import to_dense_batch

# --- CHANGE 1: Import GATv2Conv ---
from torch_geometric.nn import GATv2Conv 

# ============================================================================
# SOTA GNN Model (Cross-Attention / Docking Architecture)
# ============================================================================

from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GATv2Conv 

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim: int = 31, hidden_dims: list = [128, 128], 
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        
        # 1. Intra-Molecular Graph Layers (GATv2)
        dims = [input_dim] + hidden_dims
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Match this to your Dataset's RBF centers (You set num_centers=16)
        EDGE_DIM = 16 
        
        for i in range(len(dims) - 1):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=dims[i], 
                    out_channels=dims[i+1] // num_heads, 
                    heads=num_heads, 
                    dropout=dropout, 
                    edge_dim=EDGE_DIM,  # <--- CRITICAL: Uses your RBF distances
                    add_self_loops=False
                )
            )
            self.norms.append(nn.LayerNorm(dims[i+1]))
        
        self.graph_dim = dims[-1]
        
        # 2. Inter-Molecular Cross-Attention (The Docking Step)
        self.cross_attn_dim = self.graph_dim
        self.mhc_to_pep_attn = nn.MultiheadAttention(self.cross_attn_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.pep_to_mhc_attn = nn.MultiheadAttention(self.cross_attn_dim, num_heads=4, batch_first=True, dropout=dropout)
        
        # 3. Prediction Head
        combined_dim = 4 * self.graph_dim
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def process_subgraph(self, x, edge_index, edge_attr, batch):
        x = self.input_norm(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            # Pass the RBF edge attributes here
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x.shape[1] == x_in.shape[1]: 
                x = x + x_in # Residual
            x = norm(x)
        return x

    def forward(self, mhc, pep):
        # STEP 1: Process Graphs Independently
        mhc_feats = self.process_subgraph(mhc.x, mhc.edge_index, mhc.edge_attr, mhc.batch)
        pep_feats = self.process_subgraph(pep.x, pep.edge_index, pep.edge_attr, pep.batch)
        
        # STEP 2: Dense Batching
        mhc_dense, mhc_mask = to_dense_batch(mhc_feats, mhc.batch)
        pep_dense, pep_mask = to_dense_batch(pep_feats, pep.batch)
        
        # STEP 3: Cross-Attention (Docking)
        # Peptide searches MHC
        pep_ctx, _ = self.pep_to_mhc_attn(query=pep_dense, key=mhc_dense, value=mhc_dense, key_padding_mask=~mhc_mask)
        # MHC searches Peptide
        mhc_ctx, _ = self.mhc_to_pep_attn(query=mhc_dense, key=pep_dense, value=pep_dense, key_padding_mask=~pep_mask)
        
        # STEP 4: Smart Pooling
        def masked_pool(tensor, mask):
            tensor = tensor * mask.unsqueeze(-1).float()
            summed = tensor.sum(dim=1)
            counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            return summed / counts

        combined = torch.cat([
            masked_pool(mhc_dense, mhc_mask),
            masked_pool(pep_dense, pep_mask),
            masked_pool(mhc_ctx, mhc_mask),
            masked_pool(pep_ctx, pep_mask)
        ], dim=1)
        
        return self.fc_layers(combined).squeeze()

# ============================================================================
# Training Logic
# ============================================================================

def inverse_transform_pic50(pic50_values: np.ndarray) -> np.ndarray:
    return 10.0 ** (9.0 - pic50_values)

def train_epoch(model, loader, optimizer, criterion, device, grad_clip_norm: float = 0.0):
    model.train()
    total_loss = 0
    for i, (mhc, pep, y) in enumerate(loader):
        mhc, pep, y = mhc.to(device), pep.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(mhc, pep)
        loss = criterion(pred, y.float())
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0

def evaluate(model, loader, device):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for mhc, pep, y in loader:
            mhc, pep, y = mhc.to(device), pep.to(device), y.to(device)
            pred = model(mhc, pep)
            preds.extend(pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
    
    if not preds: return 0, 0, [], [], [], []

    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)
    
    rmse = np.sqrt(mean_squared_error(actuals_arr, preds_arr))
    r2 = r2_score(actuals_arr, preds_arr)
    ic50_preds = inverse_transform_pic50(preds_arr)
    ic50_actuals = inverse_transform_pic50(actuals_arr)
    
    return rmse, r2, preds_arr, actuals_arr, ic50_preds, ic50_actuals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()
    
    # Set seeds
    seed = CONFIG.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    logger.info("Loading Data...")
    df = pd.read_csv(CONFIG['dataset_path'])
    
    label_cols = ['IC50', 'ic50', 'binding_affinity_nM']
    label_col = next((c for c in label_cols if c in df.columns), None)
    if not label_col: raise KeyError("No IC50 column found")
    
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df = df.dropna(subset=[label_col])
    
    with open(CONFIG['mapping_file']) as f: mapping = json.load(f)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    logger.info("Initializing datasets (Lazy Loading Mode)...")
    train_ds = GraphBindingDataset(train_df, mapping, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], 
                                   cache_graphs=False, label_col=label_col)
    
    test_ds = GraphBindingDataset(test_df, mapping, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], 
                                  cache_graphs=False, label_col=label_col)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], num_workers=4)
    
    device = torch.device(CONFIG['device'])
    model = GraphAttentionNetwork(input_dim=CONFIG['input_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG.get('weight_decay', 0))
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG.get('scheduler_factor', 0.5), 
        patience=CONFIG.get('scheduler_patience', 3), min_lr=CONFIG.get('scheduler_min_lr', 1e-6)
    )
    
    logger.info(f"Starting training on {device}...")
    
    best_loss = float('inf')
    start_epoch = 0
    checkpoint_file = Path('gnn_last_checkpoint.pth')
    
    if args.resume and checkpoint_file.exists():
        try:
            ckpt = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
            best_loss = ckpt['best_loss']
            if 'scheduler_state' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception:
            start_epoch = 0
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, grad_clip_norm=CONFIG.get('grad_clip_norm', 0.0))
        val_rmse, val_r2, val_preds, val_actuals, val_ic50_preds, val_ic50_actuals = evaluate(model, test_loader, device)
        
        ic50_rmse = np.sqrt(mean_squared_error(val_ic50_actuals, val_ic50_preds))
        ic50_mae = mean_absolute_error(val_ic50_actuals, val_ic50_preds)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | pIC50 RMSE: {val_rmse:.4f} | R2: {val_r2:.4f} | LR: {current_lr:.2e}")
        logger.info(f"              | nM RMSE: {ic50_rmse:.2f} | nM MAE: {ic50_mae:.2f}")

        try:
            scheduler.step(val_rmse)
        except Exception:
            pass
        
        if val_rmse < best_loss:
            best_loss = val_rmse
            torch.save(model.state_dict(), 'best_gnn_model.pth')
            logger.info("   -> Saved new best model")
        
        try:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_loss': best_loss,
                'scheduler_state': scheduler.state_dict()
            }, checkpoint_file)
        except Exception:
            pass

if __name__ == '__main__':
    main()