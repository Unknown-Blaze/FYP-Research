"""
Graph Neural Network for MHC-Peptide Binding Affinity Prediction
"""
import warnings
# Filter specific warnings to keep logs clean
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import pandas as pd
import numpy as np
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
    'learning_rate': 0.0001,
    'hidden_dims': [64, 64],
    'num_heads': 4,
    'dropout': 0.3,
    'weight_decay': 1e-4,
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
        
        # 1. Intra-Molecular Graph Layers (GAT)
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
        
        # 2. Inter-Molecular Cross-Attention (The "Docking" Step)
        self.cross_attn_dim = self.graph_dim
        # Note: batch_first=True is required for the dense batches we create below
        self.mhc_to_pep_attn = nn.MultiheadAttention(self.cross_attn_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.pep_to_mhc_attn = nn.MultiheadAttention(self.cross_attn_dim, num_heads=4, batch_first=True, dropout=dropout)
        
        # 3. Prediction Head
        # Concatenate: [MHC_Global, Pep_Global, MHC_Interface, Pep_Interface]
        combined_dim = 4 * self.graph_dim
        
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def process_subgraph(self, x, edge_index, edge_attr, batch):
        # Run the GAT layers to get local features (Charge, Shape, etc.)
        x = self.input_norm(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            x = gat(x, edge_index, edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection if shapes match
            if x.shape == x_in.shape:
                x = x + x_in
            x = norm(x)
        return x

    def forward(self, mhc, pep):
        # STEP 1: Process MHC and Peptide separately to understand their shapes
        mhc_feats = self.process_subgraph(mhc.x, mhc.edge_index, mhc.edge_attr, mhc.batch)
        pep_feats = self.process_subgraph(pep.x, pep.edge_index, pep.edge_attr, pep.batch)
        
        # STEP 2: Prepare for Transformer Attention (Dense Batching)
        # Graphs are sparse (list of nodes), Transformers need dense tensors [Batch, Max_Nodes, Dim]
        # to_dense_batch creates these tensors and a mask (True = Real Node, False = Padding)
        mhc_dense, mhc_mask = to_dense_batch(mhc_feats, mhc.batch)
        pep_dense, pep_mask = to_dense_batch(pep_feats, pep.batch)
        
        # STEP 3: Cross-Attention (The "Docking" Simulation)
        
        # A. Peptide searches MHC: "Which MHC nodes match my charge/shape?"
        # Query = Peptide, Key/Value = MHC
        pep_context, _ = self.pep_to_mhc_attn(
            query=pep_dense, 
            key=mhc_dense, 
            value=mhc_dense, 
            key_padding_mask=~mhc_mask # Ignore padding nodes in MHC
        )
        
        # B. MHC searches Peptide
        # Query = MHC, Key/Value = Peptide
        mhc_context, _ = self.mhc_to_pep_attn(
            query=mhc_dense, 
            key=pep_dense, 
            value=pep_dense, 
            key_padding_mask=~pep_mask # Ignore padding nodes in Peptide
        )
        
        # STEP 4: Smart Pooling (Aggregating the Interaction)
        # We cannot just .mean() because of the zero-padding. We must use the mask.
        
        def masked_pool(tensor, mask):
            # Mask out padding (set to 0)
            tensor = tensor * mask.unsqueeze(-1).float()
            # Sum valid nodes
            summed = tensor.sum(dim=1)
            # Count valid nodes (clamp to avoid divide by zero)
            counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            return summed / counts

        # Pool the original features (Global Shape)
        mhc_global = masked_pool(mhc_dense, mhc_mask)
        pep_global = masked_pool(pep_dense, pep_mask)
        
        # Pool the interaction features (Binding Interface Energy)
        mhc_interface = masked_pool(mhc_context, mhc_mask)
        pep_interface = masked_pool(pep_context, pep_mask)
        
        # STEP 5: Final Prediction
        combined = torch.cat([mhc_global, pep_global, mhc_interface, pep_interface], dim=1)
        return self.fc_layers(combined).squeeze()
# ============================================================================
# Training Logic
# ============================================================================

def inverse_transform_pic50(pic50_values: np.ndarray) -> np.ndarray:
    return 10.0 ** (9.0 - pic50_values)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    # Use enumerate to avoid crash if loader is empty
    for i, (mhc, pep, y) in enumerate(loader):
        mhc, pep, y = mhc.to(device), pep.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(mhc, pep)
        loss = criterion(pred, y.float())
        loss.backward()
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
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception:
            start_epoch = 0
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_rmse, val_r2, val_preds, val_actuals, val_ic50_preds, val_ic50_actuals = evaluate(model, test_loader, device)
        
        ic50_rmse = np.sqrt(mean_squared_error(val_ic50_actuals, val_ic50_preds))
        ic50_mae = mean_absolute_error(val_ic50_actuals, val_ic50_preds)
        
        logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | pIC50 RMSE: {val_rmse:.4f} | R2: {val_r2:.4f}")
        logger.info(f"             | nM RMSE: {ic50_rmse:.2f} | nM MAE: {ic50_mae:.2f}")
        
        if val_rmse < best_loss:
            best_loss = val_rmse
            torch.save(model.state_dict(), 'best_gnn_model.pth')
            logger.info("  -> Saved new best model")
        
        try:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_loss': best_loss
            }, checkpoint_file)
        except Exception:
            pass

if __name__ == '__main__':
    main()