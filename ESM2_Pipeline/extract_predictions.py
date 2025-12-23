import sys
import os
import torch
import pandas as pd
import numpy as np
import joblib
import yaml
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

# --- Import your models ---
sys.path.append(os.path.join(os.getcwd(), 'sequence'))
sys.path.append(os.path.join(os.getcwd(), 'structure'))

try:
    from train_finetune import ESM2FineTuneModel, AppConfig
    from train_gnn_model import GraphAttentionNetwork, GraphBindingDataset
except ImportError as e:
    print(f"❌ DETAILED ERROR: {e}")  # <--- Print the actual error!
    sys.exit(1)

# --- CONFIG ---
CONFIG = {
    'data_path': './iedb_dataset.csv', # Your full dataset
    'output_csv': 'ensemble_meta_data.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 64,
    
    # Model Paths
    'esm_ckpt': 'checkpoints/checkpoints_esm2_warmstart/best_model.pt',
    'esm_config': 'sequence/config.yaml',
    'esm_scaler': 'checkpoints/checkpoints_esm2_warmstart/label_scaler.joblib',
    
    'gnn_ckpt': 'checkpoints_structure/best_gnn_model.pth',
    'gnn_mapping': './structure/row_to_pdb_map.json',
    'gnn_mhc': 'data/graphs/mhc',
    'gnn_pep': 'data/graphs/peptide'
}

def load_models(device):
    print("⏳ Loading Models...")
    
    # 1. ESM Model
    with open(CONFIG['esm_config']) as f: 
        esm_yaml = yaml.safe_load(f)
    
    # --- FIX START: Patch the config to satisfy Pydantic validation ---
    # The config expects the data file to be in a specific place relative to the 'sequence' folder.
    # We don't need the training data now, so we just point it to ANY valid file (like this script)
    # to bypass the "File not found" error.
    esm_yaml['data_file'] = __file__ 
    # --- FIX END ---

    esm_cfg = AppConfig(**esm_yaml)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(esm_cfg.tokenizer_name)
    
    esm_model = ESM2FineTuneModel(
        esm_model_name=esm_cfg.model_name,
        tokenizer_name=esm_cfg.tokenizer_name,
        num_ca_layers=esm_cfg.num_ca_layers,
        num_heads=esm_cfg.num_attention_heads,
        freeze_esm=esm_cfg.freeze_esm
    ).to(device)
    
    # Load Weights
    # We use strict=False because sometimes the model wrapper keys (module.) might differ slightly
    esm_model.load_state_dict(torch.load(CONFIG['esm_ckpt'], map_location=device), strict=False)
    esm_model.eval()
    
    # 2. ESM Scaler (to denormalize)
    try:
        scaler = joblib.load(CONFIG['esm_scaler'])
        scaler_min = torch.tensor(scaler.min_[0], device=device)
        scaler_scale = torch.tensor(scaler.scale_[0], device=device)
        print("✅ ESM Scaler loaded.")
    except:
        print("⚠️ ESM Scaler NOT found. Using raw outputs (Check this!).")
        scaler_min, scaler_scale = 0, 1

    # 3. GNN Model
    gnn_model = GraphAttentionNetwork(input_dim=31).to(device)
    ckpt = torch.load(CONFIG['gnn_ckpt'], map_location=device)
    state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    gnn_model.load_state_dict(state)
    gnn_model.eval()
    
    return esm_model, gnn_model, tokenizer, scaler_min, scaler_scale

# Add this import at the top if missing
from torch_geometric.data import Batch
from torch.utils.data import DataLoader, Dataset

# --- 1. Helper Dataset Class ---
class InferenceDataset(Dataset):
    def __init__(self, df, gnn_ds, tokenizer):
        self.df = df
        self.gnn_ds = gnn_ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load Graph (CPU intensive)
        mhc_graph, pep_graph, target = self.gnn_ds[idx]
        
        # Tokenize (CPU intensive)
        mhc_seq = row['mhc_sequence'] # Using your fixed column name
        pep_seq = row['peptide']      # Using your fixed column name
        
        return mhc_seq, pep_seq, mhc_graph, pep_graph, target, len(pep_seq), row.get('mean_plddt', 0.0)

# --- 2. Collate Function (The magic glue) ---
def collate_wrapper(batch, tokenizer):
    mhc_seqs, pep_seqs, mhc_graphs, pep_graphs, targets, lens, plddts = zip(*batch)
    
    # Batch Tokenize Sequence
    mhc_tok = tokenizer(list(mhc_seqs), return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    pep_tok = tokenizer(list(pep_seqs), return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    
    # Batch Graphs (PyG specific)
    mhc_batch_graph = Batch.from_data_list(mhc_graphs)
    pep_batch_graph = Batch.from_data_list(pep_graphs)
    
    return mhc_tok, pep_tok, mhc_batch_graph, pep_batch_graph, torch.tensor(targets), lens, plddts

# --- 3. Optimized Main Loop ---
def main():
    # Update config to higher batch size for A100
    CONFIG['batch_size'] = 256  # <--- Increased from 64 to 256
    
    device = torch.device(CONFIG['device'])
    esm_model, gnn_model, tokenizer, s_min, s_scale = load_models(device)
    
    # Load Data
    print("⏳ Reading CSV...")
    df = pd.read_csv(CONFIG['data_path']).dropna(subset=['mhc_sequence', 'peptide'])
    
    # Load GNN Dataset Wrapper
    with open(CONFIG['gnn_mapping']) as f: mapping = json.load(f)
    gnn_ds = GraphBindingDataset(df, mapping, CONFIG['gnn_mhc'], CONFIG['gnn_pep'], label_col='binding_affinity_nM')
    
    # Create DataLoader (This handles the batching)
    dataset = InferenceDataset(df, gnn_ds, tokenizer)
    loader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=16,         
        collate_fn=lambda x: collate_wrapper(x, tokenizer),
        pin_memory=True,
        # --- ADD THESE TWO LINES ---
        prefetch_factor=2,      # Ask workers to load 2 batches ahead
        persistent_workers=True # Keep workers alive (don't kill them after every batch)
    )
    
    results = []
    print(f"🚀 Running Batched Inference on {len(df)} samples with Batch Size {CONFIG['batch_size']}...")
    
    # Iterate through batches
    for batch in tqdm(loader):
        try:
            mhc_tok, pep_tok, mhc_graph, pep_graph, targets, lens, plddts = batch
            
            # Move to GPU
            esm_batch_mhc = {k: v.to(device) for k, v in mhc_tok.items()}
            esm_batch_pep = {k: v.to(device) for k, v in pep_tok.items()}
            mhc_graph = mhc_graph.to(device)
            pep_graph = pep_graph.to(device)
            
            with torch.no_grad():
                # ESM (Batch)
                esm_raw = esm_model(esm_batch_mhc, esm_batch_pep)
                esm_pred = (esm_raw - s_min) / s_scale 
                
                # GNN (Batch)
                gnn_pred = gnn_model(mhc_graph, pep_graph)
            
            # Unpack batch results
            # ESM output is [Batch, 1], GNN is [Batch] or [Batch, 1]
            esm_vals = esm_pred.squeeze().cpu().tolist()
            gnn_vals = gnn_pred.squeeze().cpu().tolist()
            target_vals = targets.tolist()
            
            # Handle case where batch_size=1 makes these floats instead of lists
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
            print(f"Batch Error: {e}")
            continue

    pd.DataFrame(results).to_csv(CONFIG['output_csv'], index=False)
    print(f"✅ Extraction Complete! Saved to {CONFIG['output_csv']}")

if __name__ == "__main__":
    main()