import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import yaml
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch

# --- 1. DEFINITIONS (MUST MATCH TRAINING) ---
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
    def __init__(self, input_dim=23, hidden_dims=[512, 512, 512], num_heads=16, dropout=0.0):
        super().__init__()
        self.dropout = dropout 
        dims = [input_dim] + hidden_dims
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        for i in range(len(dims) - 1):
            self.gat_layers.append(GATv2Conv(dims[i], dims[i+1]//num_heads, heads=num_heads, dropout=dropout, edge_dim=16))
            self.norms.append(nn.LayerNorm(dims[i+1]))
            
        self.graph_dim = dims[-1]
        self.mhc_to_pep_attn = nn.MultiheadAttention(self.graph_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.pep_to_mhc_attn = nn.MultiheadAttention(self.graph_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.embedding_dim = 4 * self.graph_dim
        self.fusion_module = MultiHeadAttentionFusion(self.embedding_dim, num_heads=num_heads)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.embedding_dim, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
    
    def process_subgraph(self, x, edge_index, edge_attr):
        x = self.input_norm(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            x_in = x
            if edge_index.numel() > 0: x = gat(x, edge_index, edge_attr=edge_attr)
            if x.shape[1] == x_in.shape[1]: x = x + x_in 
            x = F.elu(norm(x)) 
        return x

    def forward(self, mhc, pep):
        mhc_feats = self.process_subgraph(mhc.x, mhc.edge_index, mhc.edge_attr)
        pep_feats = self.process_subgraph(pep.x, pep.edge_index, pep.edge_attr)
        mhc_dense, mhc_mask = to_dense_batch(mhc_feats, mhc.batch)
        pep_dense, pep_mask = to_dense_batch(pep_feats, pep.batch)
        pep_ctx, _ = self.pep_to_mhc_attn(pep_dense, mhc_dense, mhc_dense, key_padding_mask=~mhc_mask)
        mhc_ctx, _ = self.mhc_to_pep_attn(mhc_dense, pep_dense, pep_dense, key_padding_mask=~pep_mask)
        def masked_pool(t, m): return (t * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        combined = torch.cat([masked_pool(mhc_dense, mhc_mask), masked_pool(pep_dense, pep_mask),
                              masked_pool(mhc_ctx, mhc_mask), masked_pool(pep_ctx, pep_mask)], dim=1)
        fused = self.fusion_module(combined.view(-1, 4, combined.shape[-1]))
        return self.fc_layers(fused).squeeze()

# --- 2. CONFIGURATION ---
CONFIG = {
    'data_path': './iedb_dataset.csv', 
    'output_csv': 'ensemble_meta_data.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 256, 
    
    'esm_ckpt': 'checkpoints/checkpoints_esm2_warmstart/best_model.pt',
    'esm_config': 'sequence/config.yaml',
    'esm_scaler': 'checkpoints/checkpoints_esm2_warmstart/label_scaler.joblib',
    
    'gnn_ckpt': 'gnn_a100_best.pth',
    'gnn_mapping': './structure/row_to_pdb_map.json',
    'gnn_mhc': 'data/graphs/ensemble/mhc',
    'gnn_pep': 'data/graphs/ensemble/peptide'
}

# --- 3. IMPORTS AND LOADING ---
sys.path.append(os.path.join(os.getcwd(), 'sequence'))
sys.path.append(os.path.join(os.getcwd(), 'structure'))

try:
    from train_finetune import ESM2FineTuneModel, AppConfig
    from train_gnn_model import GraphBindingDataset
except ImportError as e:
    print(f"❌ IMPORTS FAILED: {e}")
    print("Ensure 'sequence/' and 'structure/' folders exist and contain train_finetune.py / train_gnn_model.py")
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
        # NOTE: Standard unscaling is typically y_real = y_pred * scale + min
        scaler_min = torch.tensor(scaler.min_[0], device=device) if hasattr(scaler, 'min_') else 0.0
        scaler_scale = torch.tensor(scaler.scale_[0], device=device) if hasattr(scaler, 'scale_') else 1.0
        print("✅ Scaler Loaded.")
    except:
        print("⚠️ Scaler NOT found, using identity.")
        scaler_min, scaler_scale = 0.0, 1.0

    # 3. GNN Model (Dynamic Load)
    print(f"⏳ Loading GNN from {CONFIG['gnn_ckpt']}...")
    ckpt = torch.load(CONFIG['gnn_ckpt'], map_location=device)
    
    # Check if config is saved in checkpoint (from our new training script)
    saved_conf = ckpt.get('config', None)
    if saved_conf:
        print(f"✅ Found saved config in checkpoint. Hidden Dims: {saved_conf['hidden_dims']}")
        gnn_model = EnsembleGraphNetwork(
            input_dim=saved_conf['input_dim'],
            hidden_dims=saved_conf['hidden_dims'], 
            num_heads=saved_conf['num_heads'],
            dropout=0.0 # Inference dropout
        ).to(device)
    else:
        print("⚠️ No config found in checkpoint. Using hardcoded defaults.")
        gnn_model = EnsembleGraphNetwork(hidden_dims=[512, 512, 512], num_heads=16, dropout=0.0).to(device)

    # Handle state dict structure
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    gnn_model.load_state_dict(state)
    gnn_model.eval()
    
    return esm_model, gnn_model, tokenizer, scaler_min, scaler_scale

# --- 4. DATASET HELPER ---
class InferenceDataset(Dataset):
    def __init__(self, df, gnn_ds, tokenizer):
        self.df = df
        self.gnn_ds = gnn_ds
        self.tokenizer = tokenizer
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mhc_graph, pep_graph, target = self.gnn_ds[idx]
        mhc_seq = row['mhc_sequence'] 
        pep_seq = row['peptide'] 
        # Return pLDDT if column exists, else 0
        plddt = row.get('mean_plddt', 0.0) 
        if pd.isna(plddt): plddt = 0.0
        return mhc_seq, pep_seq, mhc_graph, pep_graph, target, len(pep_seq), float(plddt)

def collate_wrapper(batch, tokenizer):
    mhc_seqs, pep_seqs, mhc_graphs, pep_graphs, targets, lens, plddts = zip(*batch)
    mhc_tok = tokenizer(list(mhc_seqs), return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    pep_tok = tokenizer(list(pep_seqs), return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    mhc_batch_graph = Batch.from_data_list(mhc_graphs)
    pep_batch_graph = Batch.from_data_list(pep_graphs)
    return mhc_tok, pep_tok, mhc_batch_graph, pep_batch_graph, torch.tensor(targets), lens, plddts

# --- 5. MAIN ---
def main():
    device = torch.device(CONFIG['device'])
    esm_model, gnn_model, tokenizer, s_min, s_scale = load_models(device)
    
    print("⏳ Reading CSV...")
    df = pd.read_csv(CONFIG['data_path']).dropna(subset=['mhc_sequence', 'peptide'])
    
    with open(CONFIG['gnn_mapping']) as f: mapping = json.load(f)
    gnn_ds = GraphBindingDataset(df, mapping, CONFIG['gnn_mhc'], CONFIG['gnn_pep'], label_col='binding_affinity_nM')
    
    dataset = InferenceDataset(df, gnn_ds, tokenizer)
    loader = DataLoader(
        dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8, # Reduced workers slightly for stability
        collate_fn=lambda x: collate_wrapper(x, tokenizer), pin_memory=True
    )
    
    results = []
    print(f"🚀 Running Inference on {len(df)} samples...")
    
    for batch in tqdm(loader):
        try:
            mhc_tok, pep_tok, mhc_graph, pep_graph, targets, lens, plddts = batch
            esm_batch_mhc = {k: v.to(device) for k, v in mhc_tok.items()}
            esm_batch_pep = {k: v.to(device) for k, v in pep_tok.items()}
            mhc_graph = mhc_graph.to(device)
            pep_graph = pep_graph.to(device)
            
            with torch.no_grad():
                # ESM Inference & Unscaling
                esm_raw = esm_model(esm_batch_mhc, esm_batch_pep)
                # FIX: Standard Scaler inversion is usually (raw * scale) + min 
                # (Assuming 'raw' is normalized output and we want real values)
                # If your scaler was MinMax: (val - min)/(max-min) -> val = out * scale + min
                esm_pred = (esm_raw * s_scale) + s_min 
                
                # GNN Inference (Already in pIC50/log space due to HuberLoss training)
                gnn_pred = gnn_model(mhc_graph, pep_graph)
            
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
                    # Diff is only valid if both are in roughly the same unit (e.g. pIC50)
                    'diff': abs(esm_vals[i] - gnn_vals[i])
                })
                
        except Exception as e:
            print(f"⚠️ Batch Error: {e}")
            continue

    pd.DataFrame(results).to_csv(CONFIG['output_csv'], index=False)
    print(f"✅ Ensemble Training Data Saved to {CONFIG['output_csv']}")

if __name__ == "__main__":
    main()