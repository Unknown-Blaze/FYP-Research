import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GlobalAttention

# ============================================================================
# 1. MODEL DEFINITIONS (UPDATED TO MATCH UNIFIED GNN)
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
    def __init__(self, input_dim=6, hidden_dims=[256, 256, 256], num_heads=8, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        
        # --- 1. GNN Backbone (Unified) ---
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

        # --- 2. Interface-Aware Pooling ---
        self.pool_gate = nn.Sequential(
            nn.Linear(self.graph_dim, 1),
            nn.Sigmoid()
        )
        self.global_pool = GlobalAttention(gate_nn=self.pool_gate)

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

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            if x.shape[1] == x_in.shape[1]: x = x + x_in
            x = F.elu(norm(x))

        graph_embed = self.global_pool(x, batch)
        
        actual_batch_size = graph_embed.shape[0] // 4
        templates_view = graph_embed.view(actual_batch_size, 4, -1)
        fused_vec = self.fusion_module(templates_view)
        
        return self.fc_layers(fused_vec).squeeze()

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
CONFIG = {
    'data_path': './iedb_dataset.csv', 
    'output_csv': 'ensemble_meta_data.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 128, 
    
    'esm_ckpt': 'checkpoints/checkpoints_esm2_warmstart/best_model.pt',
    'esm_config': 'sequence/config.yaml',
    'esm_scaler': 'checkpoints/checkpoints_esm2_warmstart/label_scaler.joblib',
    
    'gnn_ckpt': './structure/gnn_a100_best.pth',
    # Note: We now point to the Unified folder
    'gnn_mhc': './structure/data/graphs/ensemble/mhc',
    'gnn_pep': './structure/data/graphs/ensemble/peptide' # (Unused but kept for class init)
}

# ============================================================================
# 3. SETUP & IMPORTS
# ============================================================================
sys.path.append(os.path.join(os.getcwd(), 'sequence'))
sys.path.append(os.path.join(os.getcwd(), 'structure'))

try:
    from train_finetune import ESM2FineTuneModel, AppConfig
    # We need the Dataset class from your training script
    from train_gnn_model import EnsembleDataset
except ImportError as e:
    print(f"❌ IMPORTS FAILED: {e}")
    sys.exit(1)

def load_models(device):
    print("⏳ Loading Models...")
    
    # 1. ESM Model
    with open(CONFIG['esm_config']) as f: esm_yaml = yaml.safe_load(f)
    esm_yaml['data_file'] = __file__ 
    esm_cfg = AppConfig(**esm_yaml)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(esm_cfg.tokenizer_name)
    esm_model = ESM2FineTuneModel(
        esm_model_name=esm_cfg.model_name, tokenizer_name=esm_cfg.tokenizer_name,
        num_ca_layers=esm_cfg.num_ca_layers, num_heads=esm_cfg.num_attention_heads,
        freeze_esm=esm_cfg.freeze_esm
    ).to(device)
    esm_model.load_state_dict(torch.load(CONFIG['esm_ckpt'], map_location=device), strict=False)
    esm_model.eval()
    
    # 2. ESM Scaler
    try:
        scaler = joblib.load(CONFIG['esm_scaler'])
        print("✅ ESM Scaler loaded.")
    except Exception:
        print("⚠️ ESM Scaler NOT found.")
        scaler = None

    # 3. GNN Model
    print(f"⏳ Loading GNN from {CONFIG['gnn_ckpt']}...")
    ckpt = torch.load(CONFIG['gnn_ckpt'], map_location=device)
    
    saved_conf = ckpt.get('config', None)
    if saved_conf:
        print(f"✅ Config: Hidden={saved_conf['hidden_dims']}, Input={saved_conf.get('input_dim', 6)}")
        gnn_model = EnsembleGraphNetwork(
            input_dim=saved_conf.get('input_dim', 6), # Default to 6 if missing
            hidden_dims=saved_conf['hidden_dims'], 
            num_heads=saved_conf['num_heads'],
            dropout=0.0
        ).to(device)
    else:
        print("⚠️ No config found. Using defaults.")
        gnn_model = EnsembleGraphNetwork().to(device)

    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    gnn_model.load_state_dict(state)
    gnn_model.eval()
    
    return esm_model, gnn_model, tokenizer, scaler

# ============================================================================
# 4. INFERENCE DATASET & COLLATE FIX (UPDATED FOR UNIFIED GRAPH)
# ============================================================================
class InferenceDataset(Dataset):
    def __init__(self, df, gnn_ds, tokenizer):
        self.df = df
        self.gnn_ds = gnn_ds
        self.tokenizer = tokenizer
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # CHANGED: Dataset now returns (Unified_Graph_Batch, Target)
        # We unpack it carefully
        graph_batch, target = self.gnn_ds[idx]
        
        mhc_seq = row['mhc_sequence'] 
        pep_seq = row['peptide'] 
        plddt = row.get('mean_plddt', 0.0) 
        if pd.isna(plddt): plddt = 0.0
        
        return mhc_seq, pep_seq, graph_batch, target, len(pep_seq), float(plddt)

def collate_wrapper(batch, tokenizer):
    # Unpack
    mhc_seqs, pep_seqs, graph_batches, targets, lens, plddts = zip(*batch)
    
    # Tokenize Sequences
    mhc_tok = tokenizer(list(mhc_seqs), return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    pep_tok = tokenizer(list(pep_seqs), return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    
    # CHANGED: Re-batch the unified graphs
    # Since 'graph_batches' are already Batch objects (containing 4 templates),
    # we use Batch.from_data_list again to merge them into a super-batch
    unified_batch_graph = Batch.from_data_list(list(graph_batches))
    
    return mhc_tok, pep_tok, unified_batch_graph, torch.tensor(targets), lens, plddts

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def main():
    device = torch.device(CONFIG['device'])
    esm_model, gnn_model, tokenizer, scaler = load_models(device)
    
    print("⏳ Reading CSV...")
    df = pd.read_csv(CONFIG['data_path']).dropna(subset=['mhc_sequence', 'peptide'])
    
    # Initialize GNN Dataset
    gnn_ds = EnsembleDataset(df, CONFIG['gnn_mhc'], CONFIG['gnn_pep'], label_col='binding_affinity_nM')
    
    dataset = InferenceDataset(df, gnn_ds, tokenizer)
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=lambda x: collate_wrapper(x, tokenizer),
        pin_memory=True
    )
    
    results = []
    print(f"🚀 Running Inference on {len(df)} samples...")
    
    for batch in tqdm(loader):
        try:
            # CHANGED: Unpack Unified Graph
            mhc_tok, pep_tok, unified_graph, targets, lens, plddts = batch
            
            esm_batch_mhc = {k: v.to(device) for k, v in mhc_tok.items()}
            esm_batch_pep = {k: v.to(device) for k, v in pep_tok.items()}
            unified_graph = unified_graph.to(device)
            
            with torch.no_grad():
                # 1. ESM Prediction
                esm_raw = esm_model(esm_batch_mhc, esm_batch_pep)
                
                # --- SCALING CHECK ---
                if scaler is not None:
                    # Inverse transform back to pIC50
                    esm_pred_np = scaler.inverse_transform(esm_raw.detach().cpu().reshape(-1, 1)).reshape(-1)
                    esm_pred = torch.from_numpy(esm_pred_np).to(device)
                else:
                    esm_pred = esm_raw.squeeze()
                
                # 2. GNN Prediction (CHANGED: Pass Unified Graph)
                gnn_pred = gnn_model(unified_graph)
            
            esm_vals = esm_pred.squeeze().cpu().tolist()
            gnn_vals = gnn_pred.squeeze().cpu().tolist()
            target_vals = targets.tolist()
            
            if not isinstance(esm_vals, list): esm_vals = [esm_vals]
            if not isinstance(gnn_vals, list): gnn_vals = [gnn_vals]
            
            for i in range(len(target_vals)):
                results.append({
                    'esm_pred': esm_vals[i],
                    'gnn_pred': gnn_vals[i],
                    'actual_pic50': target_vals[i],
                    'peptide_len': lens[i],
                    'plddt': plddts[i],
                    'diff': abs(esm_vals[i] - gnn_vals[i])
                })
                
        except Exception as e:
            print(f"⚠️ Batch Error: {e}")
            continue

    pd.DataFrame(results).to_csv(CONFIG['output_csv'], index=False)
    print(f"✅ Ensemble Training Data Saved to {CONFIG['output_csv']}")

if __name__ == "__main__":
    main()