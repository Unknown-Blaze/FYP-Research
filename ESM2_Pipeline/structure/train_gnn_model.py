"""
Graph Neural Network for MHC-Peptide Binding Affinity Prediction
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
    'input_dim': 31,  # Matches your graph node features
    'dataset_path': '../iedb_dataset.csv',
    'graphs_dir_mhc': 'data/graphs/mhc',
    'graphs_dir_peptide': 'data/graphs/peptide',
    'mapping_file': 'row_to_pdb_map.json',
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.0003,
    'hidden_dims': [64, 64],
    'num_heads': 4,
    'dropout': 0.25,
    'weight_decay': 1e-5,
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

class GraphBindingDataset(Dataset):
    def __init__(self, df, mapping, graphs_dir_mhc, graphs_dir_peptide, cache_graphs: bool = False, label_col: str = 'IC50'):
        self.df = df.copy()
        self.mapping = mapping
        self.graphs_dir_mhc = Path(graphs_dir_mhc)
        self.graphs_dir_peptide = Path(graphs_dir_peptide)
        self.cache_graphs = cache_graphs # Should be False for large datasets
        self.label_col = label_col
        self.graph_cache = {}

        if self.cache_graphs:
            logger.info("Pre-caching enabled (Warning: High RAM usage)...")
            # (Caching logic omitted for stability - sticking to Lazy Loading)

    def __len__(self):
        return len(self.df)

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
            # Check cache first, then disk
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

        # --- SAFETY CHECK (The Fix for KeyError: 'pos') ---
        # Ensure 'pos' exists even if the loaded file didn't have it
        if not hasattr(mhc_graph, 'pos') or mhc_graph.pos is None:
            mhc_graph.pos = torch.zeros((mhc_graph.num_nodes, 3), dtype=torch.float32)
            
        if not hasattr(pep_graph, 'pos') or pep_graph.pos is None:
            pep_graph.pos = torch.zeros((pep_graph.num_nodes, 3), dtype=torch.float32)

        # --- SAFETY CHECK: Ensure edge_attr exists and is correctly shaped ---
        # Many PyG graphs may omit edge attributes; create zero features if missing
        if not hasattr(mhc_graph, 'edge_attr') or mhc_graph.edge_attr is None:
            mhc_edges = mhc_graph.edge_index.shape[1] if hasattr(mhc_graph, 'edge_index') and mhc_graph.edge_index is not None else 0
            mhc_graph.edge_attr = torch.zeros((mhc_edges, 2), dtype=torch.float32)

        if not hasattr(pep_graph, 'edge_attr') or pep_graph.edge_attr is None:
            pep_edges = pep_graph.edge_index.shape[1] if hasattr(pep_graph, 'edge_index') and pep_graph.edge_index is not None else 0
            pep_graph.edge_attr = torch.zeros((pep_edges, 2), dtype=torch.float32)

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
# GNN Model
# ============================================================================

from torch_geometric.utils import to_dense_batch

class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_dim: int = 31, hidden_dims: list = [64, 64], 
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        
        # 1. Shared GAT Layers (Process both graphs with same physics)
        dims = [input_dim] + hidden_dims
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        for i in range(len(dims) - 1):
            self.gat_layers.append(
                GATConv(dims[i], dims[i+1] // num_heads, heads=num_heads, dropout=dropout)
            )
            self.norms.append(nn.LayerNorm(dims[i+1]))
        
        self.graph_dim = dims[-1]
        
        # 2. Global Pooling Interaction (The "Two-Tower" Logic)
        # We concat Mean + Max pooling for both MHC and Peptide
        # Size = (MHC_Mean + MHC_Max + Pep_Mean + Pep_Max) = 4 * Hidden
        combined_dim = 4 * self.graph_dim
        
        # 3. Interaction Head (MLP)
        # We let the MLP figure out the non-linear "binding energy"
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256), # Added BatchNorm for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 1)
        )
    
    def forward_graph(self, x, edge_index, edge_attr, batch):
        # Normalize Input
        x = self.input_norm(x)
        
        # Run GAT Layers
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            x = gat(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x.shape == x_in.shape:
                x = x + x_in # Residual
            x = norm(x)
            
        # Global Pooling (Summarize the whole molecule)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)

    def forward(self, mhc, pep):
        # 1. Get Global Summaries
        mhc_vec = self.forward_graph(mhc.x, mhc.edge_index, mhc.edge_attr, mhc.batch)
        pep_vec = self.forward_graph(pep.x, pep.edge_index, pep.edge_attr, pep.batch)
        
        # 2. Combine (Concatenate)
        # Note: Your old code multiplied (interaction = mhc * pep).
        # Concatenation is generally safer for MLP to learn interactions itself.
        combined = torch.cat([mhc_vec, pep_vec], dim=1)
        
        # 3. Predict
        return self.fc_layers(combined).squeeze()
# ============================================================================
# Training Logic
# ============================================================================

def inverse_transform_pic50(pic50_values: np.ndarray) -> np.ndarray:
    return 10.0 ** (9.0 - pic50_values)

def train_epoch(model, loader, optimizer, criterion, device, grad_clip_norm: float = 0.0):
    model.train()
    total_loss = 0
    # Use enumerate to avoid crash if loader is empty
    for i, (mhc, pep, y) in enumerate(loader):
        mhc, pep, y = mhc.to(device), pep.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(mhc, pep)
        loss = criterion(pred, y.float())
        loss.backward()
        # Gradient clipping for stability
        if grad_clip_norm and grad_clip_norm > 0:
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
    
    # Set seeds for reproducibility
    seed = CONFIG.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    logger.info("Loading Data...")
    df = pd.read_csv(CONFIG['dataset_path'])
    
    label_cols = ['IC50', 'ic50', 'binding_affinity_nM']
    label_col = next((c for c in label_cols if c in df.columns), None)
    if not label_col: raise KeyError("No IC50 column found")
    
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
    df = df.dropna(subset=[label_col])
    
    with open(CONFIG['mapping_file']) as f: mapping = json.load(f)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Lazy Loading Enabled (cache_graphs=False)
    logger.info("Initializing datasets (Lazy Loading Mode)...")
    train_ds = GraphBindingDataset(train_df, mapping, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], 
                                   cache_graphs=False, label_col=label_col)
    
    test_ds = GraphBindingDataset(test_df, mapping, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], 
                                  cache_graphs=False, label_col=label_col)
    
    # num_workers=4 speeds up disk reading
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], num_workers=4)
    
    device = torch.device(CONFIG['device'])
    model = GraphAttentionNetwork(input_dim=CONFIG['input_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG.get('weight_decay', 0))
    criterion = nn.MSELoss()

    # LR Scheduler: ReduceLROnPlateau on validation RMSE
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=CONFIG.get('scheduler_factor', 0.5),
        patience=CONFIG.get('scheduler_patience', 3),
        min_lr=CONFIG.get('scheduler_min_lr', 1e-6)
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
        logger.info(f"             | nM RMSE: {ic50_rmse:.2f} | nM MAE: {ic50_mae:.2f}")

        # Step scheduler with validation metric
        try:
            scheduler.step(val_rmse)
        except Exception:
            pass
        
        if val_rmse < best_loss:
            best_loss = val_rmse
            torch.save(model.state_dict(), 'best_gnn_model.pth')
            logger.info("  -> Saved new best model")
        
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