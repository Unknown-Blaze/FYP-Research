import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# --- Import your specific model classes ---
# (Assumes your previous files are named esm_script.py and gnn_script.py)
from esm_script import ESM2FineTuneModel, AppConfig
from gnn_script import GraphAttentionNetwork, GraphBindingDataset, CONFIG as GNN_CONFIG

# Configuration
ENSEMBLE_CONFIG = {
    'esm_checkpoint': './output_esm/best_model.pt',
    'esm_config': './esm_config.yaml',
    'esm_scaler': './output_esm/label_scaler.joblib',
    'gnn_checkpoint': './best_gnn_model.pth',
    'meta_lr': 0.001,
    'epochs': 50,
    'batch_size': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ===========================================================================
# 1. THE META-LEARNER (The Brain)
# ===========================================================================
class MetaLearner(nn.Module):
    def __init__(self, input_dim=6): # preds(2) + diff(1) + len(1) + plddt(1) + attn(1)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Final pIC50 prediction
        )
        
    def forward(self, x):
        return self.net(x)

# ===========================================================================
# 2. THE DATASET (The Glue)
# ===========================================================================
class EnsembleDataset(Dataset):
    def __init__(self, df, tokenizer, gnn_dataset):
        self.df = df
        self.tokenizer = tokenizer
        self.gnn_dataset = gnn_dataset # Reuse the complex GNN loading logic
        
        # Pre-calculate peptide lengths for speed
        self.pep_lens = df['peptide_seq'].apply(len).values
        
        # If pLDDT is in CSV, load it. If not, default to 0.
        self.plddts = df['mean_plddt'].values if 'mean_plddt' in df.columns else np.zeros(len(df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Sequence Data
        row = self.df.iloc[idx]
        mhc_seq = row['mhc_seq']
        pep_seq = row['peptide_seq']
        
        # Tokenize (ESM Input)
        mhc_tok = self.tokenizer(mhc_seq, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        pep_tok = self.tokenizer(pep_seq, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        
        esm_inputs = {
            'mhc_batch': {k: v.squeeze(0) for k, v in mhc_tok.items()},
            'pep_batch': {k: v.squeeze(0) for k, v in pep_tok.items()}
        }

        # 2. Get Structure Data (GNN Input)
        # We delegate to the existing GNN dataset class to handle graph loading
        mhc_graph, pep_graph, target = self.gnn_dataset[idx]
        
        # 3. Get Meta Features
        meta_feats = torch.tensor([
            self.pep_lens[idx],
            self.plddts[idx]
        ], dtype=torch.float32)

        return esm_inputs, mhc_graph, pep_graph, meta_feats, torch.tensor(target, dtype=torch.float32)

# ===========================================================================
# 3. THE PIPELINE WRAPPER
# ===========================================================================
class StackingPipeline(nn.Module):
    def __init__(self, esm_model, gnn_model, meta_learner, esm_scaler):
        super().__init__()
        self.esm_model = esm_model
        self.gnn_model = gnn_model
        self.meta_learner = meta_learner
        self.esm_scaler = esm_scaler # Scikit-learn scaler
        
        # FREEZE BASE MODELS
        for param in self.esm_model.parameters(): param.requires_grad = False
        for param in self.gnn_model.parameters(): param.requires_grad = False
        
        # Hook for Attention Score Extraction
        self.latest_attn_score = 0.0
        self._register_hooks()

    def _register_hooks(self):
        # Hook into the last CoAttentionLayer of ESM model to get weights
        def hook_fn(module, input, output):
            # output is (mhc_out, pep_out). We want internal attention weights.
            # Since your custom layer doesn't return weights, we approximate confidence
            # by looking at the variance of the output features (high var = specific features found)
            # Or, ideally, modify your ESM script to return attn_weights. 
            # Here, we use a proxy: Max value in the Interaction Module.
            pass 
        
        # Alternative: Just hook the interaction module output
        def interaction_hook(module, input, output):
            # If the model is confident, the interaction features should be strong (high magnitude)
            self.latest_attn_score = output.max().item()
            
        self.esm_model.interaction_module.register_forward_hook(interaction_hook)

    def forward(self, esm_inputs, mhc_graph, pep_graph, meta_static_feats):
        # 1. ESM Inference
        esm_pred_norm = self.esm_model(esm_inputs['mhc_batch'], esm_inputs['pep_batch'])
        
        # DENORMALIZE ESM PRED (Critical Step)
        # We must do this carefully inside the graph if possible, or approximate
        # Since standard scalers are CPU, we manually apply the math:
        # x_orig = x_norm * (max - min) + min
        scale = torch.tensor(self.esm_scaler.scale_[0], device=esm_pred_norm.device)
        min_val = torch.tensor(self.esm_scaler.min_[0], device=esm_pred_norm.device)
        esm_pred_real = (esm_pred_norm - min_val) / scale
        
        # 2. GNN Inference
        gnn_pred = self.gnn_model(mhc_graph, pep_graph).unsqueeze(1)
        
        # 3. Dynamic Feature Engineering
        diff = torch.abs(esm_pred_real - gnn_pred)
        attn_conf = torch.tensor([[self.latest_attn_score]], device=esm_pred_real.device)
        
        # 4. Construct Meta-Input
        # [ESM_Pred, GNN_Pred, Diff, PepLen, pLDDT, AttnConf]
        meta_input = torch.cat([
            esm_pred_real, 
            gnn_pred, 
            diff, 
            meta_static_feats, 
            attn_conf
        ], dim=1)
        
        # 5. Meta-Learner
        final_pred = self.meta_learner(meta_input)
        
        return final_pred

# ===========================================================================
# 4. TRAINING LOOP
# ===========================================================================
def train_ensemble():
    device = torch.device(ENSEMBLE_CONFIG['device'])
    
    # A. Load Base Models
    print("Loading ESM Model...")
    esm_cfg = AppConfig(**joblib.load(ENSEMBLE_CONFIG['esm_config'])) # Assuming config saved as dict or yaml
    # Note: You might need to reload config from yaml if not pickled
    esm_model = ESM2FineTuneModel(
        esm_model_name=esm_cfg.model_name, 
        tokenizer_name=esm_cfg.tokenizer_name,
        # ... pass other args from your config
    ).to(device)
    esm_model.load_state_dict(torch.load(ENSEMBLE_CONFIG['esm_checkpoint']))
    esm_model.eval()
    
    print("Loading GNN Model...")
    gnn_model = GraphAttentionNetwork(input_dim=31).to(device)
    gnn_model.load_state_dict(torch.load(ENSEMBLE_CONFIG['gnn_checkpoint'])['model_state'])
    gnn_model.eval()
    
    # B. Load Auxiliaries
    esm_scaler = joblib.load(ENSEMBLE_CONFIG['esm_scaler'])
    tokenizer = AutoTokenizer.from_pretrained(esm_cfg.tokenizer_name)
    
    # C. Data Setup
    df = pd.read_csv("your_data.csv")
    # Initialize the GNN dataset class (acts as a helper for the ensemble dataset)
    # We pass dummy mapping/dirs as examples
    base_gnn_ds = GraphBindingDataset(df, {}, './graphs/mhc', './graphs/pep')
    
    ensemble_ds = EnsembleDataset(df, tokenizer, base_gnn_ds)
    # Collate function needs to handle PyG graphs (Batching) AND PyTorch tensors
    # This requires a custom collate or careful handling. 
    # For simplicity, batch_size=1 is easiest, or use PyG DataLoader and manually stack ESM inputs
    loader = DataLoader(ensemble_ds, batch_size=1, shuffle=True) 
    
    # D. Meta-Learner Setup
    pipeline = StackingPipeline(esm_model, gnn_model, MetaLearner(), esm_scaler).to(device)
    optimizer = torch.optim.Adam(pipeline.meta_learner.parameters(), lr=ENSEMBLE_CONFIG['meta_lr'])
    criterion = nn.MSELoss()
    
    print("Starting Meta-Training...")
    for epoch in range(ENSEMBLE_CONFIG['epochs']):
        pipeline.train() # Sets meta_learner to train, but base models stay frozen via requires_grad=False
        total_loss = 0
        
        for batch in loader:
            esm_inputs, mhc_graph, pep_graph, meta_feats, target = batch
            
            # Move to device
            esm_inputs['mhc_batch'] = {k: v.to(device) for k,v in esm_inputs['mhc_batch'].items()}
            esm_inputs['pep_batch'] = {k: v.to(device) for k,v in esm_inputs['pep_batch'].items()}
            mhc_graph = mhc_graph.to(device)
            pep_graph = pep_graph.to(device)
            meta_feats = meta_feats.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            pred = pipeline(esm_inputs, mhc_graph, pep_graph, meta_feats)
            loss = criterion(pred.squeeze(), target.squeeze())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    train_ensemble()