import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
CONFIG = {
    'data_path': './iedb_dataset.csv', 
    'output_csv': 'ensemble_meta_data.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 16,
    
    # --- ESM PATHS ---
    'esm_ckpt': 'checkpoints/checkpoints_esm2_warmstart/best_model.pt',
    'esm_config': 'sequence/config.yaml',
    'esm_scaler': 'checkpoints/checkpoints_esm2_warmstart/label_scaler.joblib',
    
    # --- GNN PATHS (UPDATED) ---
    'gnn_ckpt': 'structure/checkpoints/gnn_physics_best.pth', # <--- FIXED PATH
    'gnn_mhc': 'structure/data/graphs/ensemble/mhc',
    'gnn_pep': 'structure/data/graphs/ensemble/peptide'
}

BASE_DIR = Path(__file__).resolve().parent
CONFIG['data_path'] = str(BASE_DIR / 'iedb_dataset.csv')
CONFIG['output_csv'] = str(BASE_DIR / 'ensemble_meta_data.csv')
CONFIG['esm_ckpt'] = str(BASE_DIR / 'checkpoints' / 'checkpoints_esm2_warmstart' / 'best_model.pt')
CONFIG['esm_scaler'] = str(BASE_DIR / 'checkpoints' / 'checkpoints_esm2_warmstart' / 'label_scaler.joblib')
CONFIG['gnn_ckpt'] = str(BASE_DIR / 'structure' / 'checkpoints' / 'gnn_physics_best.pth')
CONFIG['gnn_mhc'] = str(BASE_DIR / 'structure' / 'data' / 'graphs' / 'ensemble' / 'mhc')
CONFIG['gnn_pep'] = str(BASE_DIR / 'structure' / 'data' / 'graphs' / 'ensemble' / 'peptide')

# ============================================================================
# 2. SETUP & IMPORTS
# ============================================================================
sys.path.append(str(BASE_DIR / 'sequence'))
sys.path.append(str(BASE_DIR / 'structure'))

try:
    from train_finetune import ESM2FineTuneModel, AppConfig
    
    # IMPORT GNN MODEL & DATASET FROM YOUR TRAINING SCRIPT
    # This ensures dimensions match exactly what you trained with
    from train_gnn_model import EnsembleGraphNetwork, EnsembleDataset, CONFIG as GNN_TRAIN_CONFIG
    
except ImportError as e:
    print(f"❌ IMPORTS FAILED: {e}")
    print("Ensure you are running this from the project root and paths are correct.")
    sys.exit(1)

def load_models(device):
    print("⏳ Loading Models...")
    
    # --- LOAD ESM ---
    with open(BASE_DIR / CONFIG['esm_config']) as f: esm_yaml = yaml.safe_load(f)
    esm_yaml['data_file'] = str(BASE_DIR / 'iedb_dataset.csv')
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
    
    try:
        scaler = joblib.load(CONFIG['esm_scaler'])
        print("✅ ESM Scaler loaded.")
    except Exception:
        print("⚠️ ESM Scaler NOT found. Using raw output.")
        scaler = None

    # --- LOAD GNN ---
    print(f"⏳ Loading GNN from {CONFIG['gnn_ckpt']}...")
    
    # Initialize with dimensions from the Training Config to be safe
    # But hardcoding here to be explicit about what we expect
    gnn_model = EnsembleGraphNetwork(
        input_dim=30,   # <--- FIXED (Physics Dims)
        edge_dim=18,    # <--- FIXED (Physics Dims)
        hidden_dims=[512, 512, 512],
        num_heads=16,
        dropout=0.0
    ).to(device)

    # Load Weights
    ckpt = torch.load(CONFIG['gnn_ckpt'], map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    gnn_model.load_state_dict(state)
    gnn_model.eval()
    
    return esm_model, gnn_model, tokenizer, scaler

# ============================================================================
# 3. INFERENCE DATASET & COLLATE
# ============================================================================
class InferenceDataset(Dataset):
    def __init__(self, df, gnn_ds):
        self.df = df
        self.gnn_ds = gnn_ds
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get Graph (Uses the Self-Healing logic from train_gnn_model.py)
        graph_batch, target = self.gnn_ds[idx]
        
        mhc_seq = row['mhc_sequence'] 
        pep_seq = row['peptide'] 
        plddt = row.get('mean_plddt', 0.0) 
        if pd.isna(plddt): plddt = 0.0
        
        return mhc_seq, pep_seq, graph_batch, target, len(pep_seq), float(plddt)

def collate_wrapper(batch, tokenizer):
    mhc_seqs, pep_seqs, graph_batches, targets, lens, plddts = zip(*batch)
    
    mhc_tok = tokenizer(list(mhc_seqs), return_tensors='pt', padding=True, truncation=True, max_length=512)
    pep_tok = tokenizer(list(pep_seqs), return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    unified_batch_graph = Batch.from_data_list(list(graph_batches))
    
    return mhc_tok, pep_tok, unified_batch_graph, torch.tensor(targets), lens, plddts

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================
def main():
    device = torch.device(CONFIG['device'])
    use_cuda = device.type == 'cuda'
    esm_model, gnn_model, tokenizer, scaler = load_models(device)
    
    print("⏳ Reading CSV...")
    df = pd.read_csv(CONFIG['data_path']).dropna(subset=['mhc_sequence', 'peptide'])
    
    # Initialize the Robust Dataset from Structure Module
    gnn_ds = EnsembleDataset(df, CONFIG['gnn_mhc'], CONFIG['gnn_pep'], label_col='binding_affinity_nM')
    
    dataset = InferenceDataset(df, gnn_ds)
    loader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=4, 
        collate_fn=lambda x: collate_wrapper(x, tokenizer),
        pin_memory=use_cuda
    )
    
    results = []
    print(f"🚀 Running Inference on {len(df)} samples...")
    
    for batch in tqdm(loader):
        try:
            mhc_tok, pep_tok, unified_graph, targets, lens, plddts = batch
            
            esm_batch_mhc = {k: v.to(device) for k, v in mhc_tok.items()}
            esm_batch_pep = {k: v.to(device) for k, v in pep_tok.items()}
            unified_graph = unified_graph.to(device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=use_cuda):
                    esm_raw = esm_model(esm_batch_mhc, esm_batch_pep)
                    gnn_pred = gnn_model(unified_graph)
                
                # ESM Post-Process
                if scaler is not None:
                    esm_raw_cpu = esm_raw.float().cpu().numpy().reshape(-1, 1)
                    esm_pred_np = scaler.inverse_transform(esm_raw_cpu).reshape(-1)
                    esm_vals = esm_pred_np.tolist()
                else:
                    esm_vals = esm_raw.float().squeeze().cpu().tolist()
                
                gnn_vals = gnn_pred.float().squeeze().cpu().tolist()
            
            target_vals = targets.tolist()
            
            if isinstance(esm_vals, float): esm_vals = [esm_vals]
            if isinstance(gnn_vals, float): gnn_vals = [gnn_vals]
            
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