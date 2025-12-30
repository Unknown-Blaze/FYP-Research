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
# --- FIX 1: Added GlobalAttention import ---
from torch_geometric.nn import GATv2Conv, GlobalAttention
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
from tqdm import tqdm

# ============================================================================
# 1. Configuration (A100 TUNED)
# ============================================================================

CONFIG = {
    'input_dim': 6,  
    'dataset_path': '../iedb_dataset.csv',
    'graphs_dir_mhc': 'data/graphs/ensemble/mhc',
    'graphs_dir_peptide': 'data/graphs/ensemble/peptide',
    
    # --- NEW: ORGANIZATION PATHS ---
    'logs_dir': 'logs',           # <--- New folder for logs
    'checkpoints_dir': 'checkpoints', # <--- New folder for models
    
    'batch_size': 32,
    'grad_accum_steps': 16,          
    'num_epochs': 150,        
    'learning_rate': 3e-4,        
    'warmup_epochs': 10,          
    'hidden_dims': [512, 512, 512],
    'num_heads': 16,                
    'dropout': 0.3,               
    'weight_decay': 1e-4,         
    'seed': 42,
    'grad_clip_norm': 1.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Create directories immediately
Path(CONFIG['logs_dir']).mkdir(parents=True, exist_ok=True)
Path(CONFIG['checkpoints_dir']).mkdir(parents=True, exist_ok=True)

# ============================================================================
# 0. Global Setup & Logging
# ============================================================================

USE_WANDB = False 

if USE_WANDB:
    import wandb

# CHANGE: Path joins the directory with the filename
log_file = Path(CONFIG['logs_dir']) / f"gnn_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
# 2. Dataset & Helper Functions
# ============================================================================

def rbf_expansion(distances, num_centers=16, cutoff=10.0):
    centers = torch.linspace(0, cutoff, num_centers, device=distances.device)
    width = centers[1] - centers[0]
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / width)

class EnsembleDataset(Dataset):
    def __init__(self, df, graphs_dir_mhc, graphs_dir_peptide, label_col='binding_affinity_nM'):
        self.df = df.copy()
        self.graphs_dir = Path(graphs_dir_mhc) 
        self.label_col = label_col if label_col in self.df.columns else 'IC50'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        graph_list = []
        
        for t in range(4):
            file_id = f"{row.name}_T{t}.pt"
            path = self.graphs_dir / file_id
            
            try:
                # 1. LOAD GRAPH
                graph = torch.load(path)
                
                # 2. EXTRACT SIMILARITY SAFELY
                if hasattr(graph, 'identity'):
                    sim_val = float(graph.identity)
                else:
                    sim_val = 0.0 # Default/Fallback
                
                # --- NEW: Attach as a Tensor Property ---
                # This ensures PyTorch stacks it into [Batch, 1] automatically
                graph.sim_score = torch.tensor([sim_val], dtype=torch.float32)

                # 3. CLEANUP
                if hasattr(graph, 'y'): del graph.y
                if hasattr(graph, 'identity'): del graph.identity # Remove original key

                num_nodes = graph.x.size(0)
                # Fix edges
                if graph.edge_index.numel() > 0:
                    mask = (graph.edge_index[0] < num_nodes) & (graph.edge_index[1] < num_nodes)
                    graph.edge_index = graph.edge_index[:, mask]
                    if graph.edge_attr is not None:
                        graph.edge_attr = graph.edge_attr[mask]

                # Fix mask
                if hasattr(graph, 'chain_mask'):
                    graph.chain_mask = graph.chain_mask.bool()
                else:
                    graph.chain_mask = torch.ones(num_nodes, dtype=torch.bool)
                
            except (FileNotFoundError, Exception):
                # Dummy Graph
                dummy_x = torch.zeros((1, CONFIG['input_dim']), dtype=torch.float32)
                dummy_pos = torch.zeros((1, 3), dtype=torch.float32)
                dummy_edge = torch.zeros((2, 0), dtype=torch.long)
                dummy_attr = torch.zeros((0, 16), dtype=torch.float32)
                
                graph = Data(x=dummy_x, pos=dummy_pos, edge_index=dummy_edge, 
                             edge_attr=dummy_attr, batch=torch.zeros(1).long(),
                             chain_mask=torch.zeros(1, dtype=torch.bool))
                
                # Attach Dummy Score
                graph.sim_score = torch.tensor([0.0], dtype=torch.float32)

            graph_list.append(graph)

        # Create Batch
        batch = Batch.from_data_list(graph_list)
        
        try:
            val = float(row[self.label_col])
            val = min(val, 50000.0)
            target = -np.log10(val * 1e-9 + 1e-12) 
        except:
            target = 0.0

        return batch, float(target)
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
    def __init__(self, input_dim=6, hidden_dims=[256, 256, 256], num_heads=8, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        # --- 1. GNN Backbone ---
        dims = [input_dim] + hidden_dims
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.gat_layers.append(
                GATv2Conv(dims[i], dims[i+1]//num_heads, heads=num_heads, 
                          dropout=dropout, edge_dim=16, concat=True)
            )
            self.norms.append(nn.LayerNorm(dims[i+1]))
            
        self.graph_dim = dims[-1]

        # --- 2. Pooling ---
        self.pool_gate = nn.Sequential(
            nn.Linear(self.graph_dim, 1),
            nn.Sigmoid()
        )
        self.global_pool = GlobalAttention(gate_nn=self.pool_gate)

        # --- NEW: Similarity Injection ---
        # We will concat the graph embedding (512) with the similarity score (1)
        # Then project back to 512
        self.sim_proj = nn.Sequential(
            nn.Linear(self.graph_dim + 1, self.graph_dim),
            nn.LayerNorm(self.graph_dim),
            nn.ReLU()
        )

        # --- 3. Template Fusion ---
        self.fusion_module = MultiHeadAttentionFusion(self.graph_dim, num_heads=num_heads)

        # --- 4. Prediction Head ---
        self.fc_layers = nn.Sequential(
            nn.Linear(self.graph_dim, 128),
            nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # --- UPGRADE 1: THE BIAS TRICK ---
        # Initialize final bias to 6.0 (approx mean pIC50)
        self.fc_layers[-1].bias.data.fill_(6.0)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Capture similarity scores
        sim_scores = data.sim_score 

        # 1. GNN
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            if x.shape[1] == x_in.shape[1]: x = x + x_in
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Pooling -> [Batch*4, 512]
        graph_embed = self.global_pool(x, batch)
        
        # --- FIX: Ensure dimensions match for concatenation ---
        if sim_scores.dim() == 1:
            sim_scores = sim_scores.unsqueeze(1)
        # ----------------------------------------------------

        # --- UPGRADE 2: INJECT SIMILARITY ---
        combined = torch.cat([graph_embed, sim_scores], dim=1) 
        graph_embed = self.sim_proj(combined)
        # ------------------------------------

        # 3. Fusion
        actual_batch_size = graph_embed.shape[0] // 4
        templates_view = graph_embed.view(actual_batch_size, 4, -1)
        fused_vec = self.fusion_module(templates_view)
        
        return self.fc_layers(fused_vec).squeeze()
    
# ============================================================================
# 4. Optimized Training Loop
# ============================================================================

def train_epoch_amp(model, loader, optimizer, criterion, device, scaler, grad_clip_norm):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="Training", leave=False, dynamic_ncols=True)
    accum_steps = CONFIG.get('grad_accum_steps', 1)
    
    # Correct: Unpack only 2 items
    for i, (data, y) in enumerate(pbar):
        data = data.to(device)
        y = y.to(device)
        
        num_nodes = data.x.size(0)
        if data.edge_index.max() >= num_nodes:
            print(f"Skipping Bad Batch! Max edge index {data.edge_index.max()} >= Num nodes {num_nodes}")
            continue
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred = model(data)
            loss = criterion(pred, y.float())
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f"{loss.item() * accum_steps:.4f}"})

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, actuals = [], []
    all_sims = [] # To store identity scores

    with torch.no_grad():
        for data, y in loader:
            data = data.to(device)
            y = y.to(device)
            
            pred = model(data)
            
            preds.extend(pred.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            
            # EXTRACT SIMILARITIES FOR LOGGING
            if hasattr(data, 'similarities'):
                # Reshape flattened [Batch*4] back to [Batch, 4]
                sims = data.similarities.view(-1, 4).cpu().numpy()
                all_sims.append(sims)
    
    if not preds: return 0, 0, 0, 0
    
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    r2 = r2_score(actuals, preds)
    pcc, _ = pearsonr(actuals, preds)
    scc, _ = spearmanr(actuals, preds)
    
    # PRINT IDENTITY STATS
    if all_sims:
        all_sims = np.vstack(all_sims)
        avg_sim = np.mean(all_sims, axis=0) * 100
        print(f"\n[Validation] Avg Template Identity: T0:{avg_sim[0]:.1f}% | T1:{avg_sim[1]:.1f}% | T2:{avg_sim[2]:.1f}% | T3:{avg_sim[3]:.1f}%")

    return rmse, r2, pcc, scc

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    
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
    
    checkpoint_best = Path(CONFIG['checkpoints_dir']) / "gnn_a100_best.pth"
    checkpoint_last = Path(CONFIG['checkpoints_dir']) / "gnn_a100_last.pth"

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

        log_msg = (f"Ep {epoch+1:03d} | LR {current_lr:.2e} | Train Loss {loss:.4f} | "
                   f"Val RMSE {val_rmse:.4f} | Val R2 {val_r2:.4f} | Val PCC {val_pcc:.4f}")
        logger.info(log_msg)
        
        if USE_WANDB:
            wandb.log({
                "epoch": epoch+1, "lr": current_lr, "train_loss": loss,
                "val_rmse": val_rmse, "val_r2": val_r2, "val_pcc": val_pcc, "val_scc": val_scc
            })

        state_dict = {
            'epoch': epoch,
            'config': CONFIG,
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
            torch.save(state_dict, checkpoint_best)
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