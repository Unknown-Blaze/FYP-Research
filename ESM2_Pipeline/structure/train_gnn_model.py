"""
Graph Neural Network for MHC-Peptide Binding Affinity Prediction
(Multi-Template Ensemble Architecture - MIL-GNN)
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data, Batch 
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from datetime import datetime
import sys

# Setup logging - Tries to force immediate flush
log_file = f"gnn_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout) # Explicitly log to console too
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. Configuration 
# ============================================================================

CONFIG = {
    'input_dim': 23,  
    'dataset_path': '../iedb_dataset.csv',
    'graphs_dir_mhc': 'data/graphs/ensemble/mhc',
    'graphs_dir_peptide': 'data/graphs/ensemble/peptide',
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.00005,
    'hidden_dims': [128, 128],
    'num_heads': 4,
    'dropout': 0.3,
    'weight_decay': 1e-4,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# 2. Dataset 
# ============================================================================

def rbf_expansion(distances, num_centers=16, cutoff=10.0):
    centers = torch.linspace(0, cutoff, num_centers, device=distances.device)
    width = centers[1] - centers[0]
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / width)

class EnsembleDataset(Dataset):
    def __init__(self, df, graphs_dir_mhc, graphs_dir_peptide, label_col='binding_affinity_nM'):
        self.df = df.copy()
        self.graphs_dir_mhc = Path(graphs_dir_mhc)
        self.graphs_dir_peptide = Path(graphs_dir_peptide)
        if label_col not in self.df.columns and 'IC50' in self.df.columns:
            self.label_col = 'IC50'
        else:
            self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def _add_distance_features(self, graph):
        if graph.edge_index.numel() > 0:
            row, col = graph.edge_index
            coord_diff = graph.pos[row] - graph.pos[col]
            radial_dist = torch.norm(coord_diff, dim=-1)
            graph.edge_attr = rbf_expansion(radial_dist, num_centers=16)
        else:
            graph.edge_attr = torch.zeros((0, 16), dtype=torch.float32)
        return graph

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mhc_list = []
        pep_list = []
        
        for t in range(4):
            file_id = f"{row.name}_T{t}.pt"
            mhc_path = self.graphs_dir_mhc / file_id
            pep_path = self.graphs_dir_peptide / file_id
            
            mhc_g, pep_g = None, None
            if mhc_path.exists() and pep_path.exists():
                try:
                    mhc_g = torch.load(mhc_path)
                    pep_g = torch.load(pep_path)
                    mhc_g = self._add_distance_features(mhc_g)
                    pep_g = self._add_distance_features(pep_g)
                except:
                    pass

            if mhc_g is None or pep_g is None:
                dummy_x = torch.zeros((1, CONFIG['input_dim']), dtype=torch.float32)
                dummy_pos = torch.zeros((1, 3), dtype=torch.float32)
                dummy_edge = torch.zeros((2, 0), dtype=torch.long)
                dummy_attr = torch.zeros((0, 16), dtype=torch.float32)
                mhc_g = Data(x=dummy_x, pos=dummy_pos, edge_index=dummy_edge, edge_attr=dummy_attr)
                pep_g = Data(x=dummy_x, pos=dummy_pos, edge_index=dummy_edge, edge_attr=dummy_attr, y=torch.tensor([0.0]))

            mhc_list.append(mhc_g)
            pep_list.append(pep_g)

        mhc_batch = Batch.from_data_list(mhc_list)
        pep_batch = Batch.from_data_list(pep_list)

        try:
            val = float(row[self.label_col])
            val = min(val, 50000.0)
            target = -np.log10(val * 1e-9 + 1e-12)
        except:
            target = 0.0

        return mhc_batch, pep_batch, float(target)

# ============================================================================
# 3. Model
# ============================================================================

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, template_embeddings):
        batch_size = template_embeddings.size(0)
        query = self.query_token.expand(batch_size, -1, -1)
        attn_output, weights = self.attn(query, template_embeddings, template_embeddings)
        return self.norm(attn_output + query).squeeze(1)

class EnsembleGraphNetwork(nn.Module):
    def __init__(self, input_dim=23, hidden_dims=[128, 128], num_heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        dims = [input_dim] + hidden_dims
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        for i in range(len(dims) - 1):
            self.gat_layers.append(
                GATv2Conv(dims[i], dims[i+1]//num_heads, heads=num_heads, dropout=dropout, edge_dim=16)
            )
            self.norms.append(nn.LayerNorm(dims[i+1]))
            
        self.graph_dim = dims[-1]
        self.mhc_to_pep_attn = nn.MultiheadAttention(self.graph_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.pep_to_mhc_attn = nn.MultiheadAttention(self.graph_dim, num_heads=4, batch_first=True, dropout=dropout)
        self.embedding_dim = 4 * self.graph_dim
        self.fusion_module = MultiHeadAttentionFusion(self.embedding_dim, num_heads=4)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def process_subgraph(self, x, edge_index, edge_attr, batch):
        x = self.input_norm(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            if edge_index.numel() > 0:
                x = gat(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            if x.shape[1] == x_in.shape[1]: x = x + x_in 
            x = norm(x)
        return x

    def forward(self, mhc, pep):
        mhc_feats = self.process_subgraph(mhc.x, mhc.edge_index, mhc.edge_attr, mhc.batch)
        pep_feats = self.process_subgraph(pep.x, pep.edge_index, pep.edge_attr, pep.batch)
        
        mhc_dense, mhc_mask = to_dense_batch(mhc_feats, mhc.batch)
        pep_dense, pep_mask = to_dense_batch(pep_feats, pep.batch)
        
        pep_ctx, _ = self.pep_to_mhc_attn(pep_dense, mhc_dense, mhc_dense, key_padding_mask=~mhc_mask)
        mhc_ctx, _ = self.mhc_to_pep_attn(mhc_dense, pep_dense, pep_dense, key_padding_mask=~pep_mask)
        
        def masked_pool(tensor, mask):
            return (tensor * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        combined = torch.cat([
            masked_pool(mhc_dense, mhc_mask),
            masked_pool(pep_dense, pep_mask),
            masked_pool(mhc_ctx, mhc_mask),
            masked_pool(pep_ctx, pep_mask)
        ], dim=1)
        
        actual_batch_size = combined.shape[0] // 4
        templates_view = combined.view(actual_batch_size, 4, -1)
        fused_vec = self.fusion_module(templates_view)
        
        return self.fc_layers(fused_vec).squeeze()

# ============================================================================
# 4. Main Loop
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for i, (mhc, pep, y) in enumerate(loader):
        mhc, pep, y = mhc.to(device), pep.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(mhc, pep)
        loss = criterion(pred, y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # --- LOG PROGRESS EVERY 100 STEPS ---
        if i % 100 == 0:
            logger.info(f"   Step {i}/{len(loader)} | Batch Loss: {loss.item():.4f}")

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
    
    if not preds: return 0, 0
    return np.sqrt(mean_squared_error(actuals, preds)), r2_score(actuals, preds)

def main():
    logger.info("Loading Data...")
    df = pd.read_csv(CONFIG['dataset_path'])
    label_col = 'binding_affinity_nM' if 'binding_affinity_nM' in df.columns else 'IC50'
    df = df.dropna(subset=[label_col])
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=CONFIG['seed'])
    train_ds = EnsembleDataset(train_df, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], label_col)
    test_ds = EnsembleDataset(test_df, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], label_col)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], num_workers=4)
    
    device = torch.device(CONFIG['device'])
    model = EnsembleGraphNetwork(input_dim=CONFIG['input_dim']).to(device)
    
    # --- RESUME LOGIC ---
    checkpoint_path = "gnn_ensemble_latest.pth"
    if Path(checkpoint_path).exists():
        logger.info(f"✅ FOUND CHECKPOINT. Loading weights from {checkpoint_path}...")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
            logger.info("✅ Weights loaded successfully. Resuming training...")
        except Exception as e:
            logger.warning(f"⚠️ Could not load checkpoint: {e}. Starting from scratch.")
    else:
        logger.info("⚠️ No checkpoint found. Starting fresh.")
    # --------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.MSELoss()
    
    # --- NEW: ADD SCHEDULER ---
    # Monitors 'val_loss'. If it doesn't improve for 3 epochs, it cuts LR by half.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    # --------------------------

    logger.info("Starting training...")
    for epoch in range(CONFIG['num_epochs']):
        # Get current LR for logging
        current_lr = optimizer.param_groups[0]['lr']
        
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_rmse, val_r2 = evaluate(model, test_loader, device)
        
        # Calculate Validation Loss for the Scheduler (Approximation using RMSE^2)
        val_loss = val_rmse**2
        
        # Update Scheduler
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}: LR {current_lr:.6f} | Loss {loss:.4f} | Val RMSE {val_rmse:.4f} | Val R2 {val_r2:.4f}")
        torch.save(model.state_dict(), "gnn_ensemble_latest.pth")

if __name__ == '__main__':
    main()