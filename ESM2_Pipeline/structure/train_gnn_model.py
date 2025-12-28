"""
Graph Neural Network for MHC-Peptide Binding Affinity Prediction
(Multi-Template Ensemble Architecture - MIL-GNN)
TARGET: NVIDIA A100 OPTIMIZED (512-dim, 16 heads)
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import random
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
from scipy.stats import pearsonr, spearmanr
import logging
from datetime import datetime
import sys
import json
import gc
from tqdm import tqdm  # <--- NEW: For progress bars

# ============================================================================
# 0. Global Setup & Logging
# ============================================================================

# Toggle this for professional experiment tracking
USE_WANDB = False 

if USE_WANDB:
    import wandb

log_file = f"gnn_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 1. Configuration (A100 TUNED)
# ============================================================================

CONFIG = {
    'input_dim': 23,  
    'dataset_path': '../iedb_dataset.csv',
    'graphs_dir_mhc': 'data/graphs/ensemble/mhc',
    'graphs_dir_peptide': 'data/graphs/ensemble/peptide',
    
    # --- A100 OPTIMIZATIONS ---
    'batch_size': 512,           
    'num_epochs': 150,        
    'learning_rate': 3e-4,        
    'warmup_epochs': 10,          
    
    # --- ARCHITECTURE ---
    'hidden_dims': [512, 512, 512],
    'num_heads': 16,                
    'dropout': 0.3,               
    'weight_decay': 1e-4,         
    'seed': 42,
    'grad_clip_norm': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ============================================================================
# 2. Dataset & Helper Functions
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
        
        # Smart label detection
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
            try:
                mhc_g = torch.load(mhc_path)
                pep_g = torch.load(pep_path)
                
                if hasattr(mhc_g, 'y'): del mhc_g.y
                if hasattr(pep_g, 'y'): del pep_g.y
                
                mhc_g = self._add_distance_features(mhc_g)
                pep_g = self._add_distance_features(pep_g)
            except (FileNotFoundError, Exception):
                pass

            if mhc_g is None or pep_g is None:
                # Optimized dummy generation
                dummy_x = torch.zeros((1, CONFIG['input_dim']), dtype=torch.float32)
                dummy_pos = torch.zeros((1, 3), dtype=torch.float32)
                dummy_edge = torch.zeros((2, 0), dtype=torch.long)
                dummy_attr = torch.zeros((0, 16), dtype=torch.float32)
                mhc_g = Data(x=dummy_x, pos=dummy_pos, edge_index=dummy_edge, edge_attr=dummy_attr)
                pep_g = Data(x=dummy_x, pos=dummy_pos, edge_index=dummy_edge, edge_attr=dummy_attr)

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
        attn_output, _ = self.attn(query, template_embeddings, template_embeddings)
        return self.norm(attn_output + query).squeeze(1)

class EnsembleGraphNetwork(nn.Module):
    def __init__(self, input_dim=23, hidden_dims=[512, 512, 512], num_heads=16, dropout=0.3):
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
        
        self.mhc_to_pep_attn = nn.MultiheadAttention(self.graph_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.pep_to_mhc_attn = nn.MultiheadAttention(self.graph_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.embedding_dim = 4 * self.graph_dim
        self.fusion_module = MultiHeadAttentionFusion(self.embedding_dim, num_heads=num_heads)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024),
            nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
    
    def process_subgraph(self, x, edge_index, edge_attr, batch):
        x = self.input_norm(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            if edge_index.numel() > 0:
                x = gat(x, edge_index, edge_attr=edge_attr)
            
            # Residual if shapes match
            if x.shape[1] == x_in.shape[1]: 
                x = x + x_in 
            
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
        
        # Reshape for fusion (Batch * 4 -> Batch, 4, Feats)
        actual_batch_size = combined.shape[0] // 4
        templates_view = combined.view(actual_batch_size, 4, -1)
        fused_vec = self.fusion_module(templates_view)
        
        return self.fc_layers(fused_vec).squeeze()

# ============================================================================
# 4. Optimized Training Loop
# ============================================================================

def train_epoch_amp(model, loader, optimizer, criterion, device, scaler, grad_clip_norm):
    model.train()
    total_loss = 0
    
    # NEW: TQDM Progress Bar
    pbar = tqdm(loader, desc="Training", leave=False, dynamic_ncols=True)
    
    for mhc, pep, y in pbar:
        mhc, pep, y = mhc.to(device), pep.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred = model(mhc, pep)
            loss = criterion(pred, y.float())
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        if USE_WANDB:
            wandb.log({"train_step_loss": loss.item()})

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
    
    if not preds: return 0, 0, 0, 0
    
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    pcc, _ = pearsonr(actuals, preds)
    scc, _ = spearmanr(actuals, preds)
    
    return rmse, r2, pcc, scc

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system') # <--- ADD THIS LINE
    
    # --- Reproducibility Setup ---
    seed = CONFIG['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True 
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)

    if USE_WANDB:
        wandb.init(project="MHC-GNN-Ensemble", config=CONFIG, name=f"run_{datetime.now().strftime('%m%d_%H%M')}")

    logger.info("Initializing Data...")
    df = pd.read_csv(CONFIG['dataset_path'])
    label_col = 'binding_affinity_nM' if 'binding_affinity_nM' in df.columns else 'IC50'
    df = df.dropna(subset=[label_col])
    
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=CONFIG['seed'])
    train_ds = EnsembleDataset(train_df, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], label_col)
    test_ds = EnsembleDataset(test_df, CONFIG['graphs_dir_mhc'], CONFIG['graphs_dir_peptide'], label_col)
    
    logger.info(f"Train Size: {len(train_df)} | Test Size: {len(test_df)}")
    
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g_seed 
    )
    test_loader = DataLoader(
        test_ds, batch_size=CONFIG['batch_size'], num_workers=4, 
        pin_memory=True, persistent_workers=True,
        worker_init_fn=seed_worker, generator=g_seed
    )
    
    device = torch.device(CONFIG['device'])
    model = EnsembleGraphNetwork(
        input_dim=CONFIG['input_dim'],
        hidden_dims=CONFIG['hidden_dims'], 
        num_heads=CONFIG['num_heads'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    checkpoint_best = "gnn_a100_best.pth"
    checkpoint_last = "gnn_a100_last.pth"

    # Auto-detect device for AMP (fixes CUDA-only issue)
    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.HuberLoss(delta=1.0) 
    
    def lr_lambda(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return float(epoch + 1) / float(CONFIG['warmup_epochs'])
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # --- RESUMPTION LOGIC ---
    start_epoch = 0
    best_val_rmse = float('inf')
    early_stop_counter = 0

    if Path(checkpoint_last).exists():
        logger.info(f"🔄 Found checkpoint: {checkpoint_last}. Resuming...")
        checkpoint = torch.load(checkpoint_last)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'plateau_scheduler' in checkpoint:
            plateau_scheduler.load_state_dict(checkpoint['plateau_scheduler'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
        early_stop_counter = checkpoint.get('early_stop_counter', 0)
        
        # Fast-forward warmup
        if start_epoch > CONFIG['warmup_epochs']:
            for _ in range(CONFIG['warmup_epochs']): warmup_scheduler.step()
        
        logger.info(f"⏩ Resumed at Epoch {start_epoch} (Best RMSE: {best_val_rmse:.4f})")
    else:
        logger.info("🚀 Starting FRESH training on A100...")

    patience = 20

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        
        gc.collect()
        torch.cuda.empty_cache()
        
        loss = train_epoch_amp(model, train_loader, optimizer, criterion, device, scaler, CONFIG['grad_clip_norm'])
        val_rmse, val_r2, val_pcc, val_scc = evaluate(model, test_loader, device)
        
        if epoch < CONFIG['warmup_epochs']:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_rmse)

        # Clean Log Line
        log_msg = (f"Ep {epoch+1:03d} | LR {current_lr:.2e} | Train Loss {loss:.4f} | "
                   f"Val RMSE {val_rmse:.4f} | Val R2 {val_r2:.4f} | Val PCC {val_pcc:.4f}")
        logger.info(log_msg)
        
        if USE_WANDB:
            wandb.log({
                "epoch": epoch+1, "lr": current_lr, "train_loss": loss,
                "val_rmse": val_rmse, "val_r2": val_r2, "val_pcc": val_pcc, "val_scc": val_scc
            })

        # Save Logic
        state_dict = {
            'epoch': epoch,
            'config': CONFIG, # <--- CRITICAL: SAVE CONFIG FOR INFERENCE
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'plateau_scheduler': plateau_scheduler.state_dict(),
            'best_val_rmse': best_val_rmse,
            'early_stop_counter': early_stop_counter
        }

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            early_stop_counter = 0
            torch.save(state_dict, checkpoint_best) # Save best with full state (safe)
            logger.info(f"⭐ New Best Model Saved")
        else:
            early_stop_counter += 1
            logger.info(f"⏳ Patience: {early_stop_counter}/{patience}")

        torch.save(state_dict, checkpoint_last)

        if early_stop_counter >= patience:
            logger.info("🛑 Early Stopping Triggered.")
            break

if __name__ == '__main__':
    main()