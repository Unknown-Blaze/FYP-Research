"""
ESM-2 Fine-tuning Version

This version unfreezes ESM-2 weights to allow the entire model to learn.
This should give MUCH better performance but trains slower.

Key difference: ESM-2 weights are trainable (not frozen).
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmModel
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class ESM2ConvNeXtModel(nn.Module):
    """
    ESM-2 + ConvNeXt fusion for MHC-peptide binding prediction
    
    KEY CHANGE: ESM-2 weights are NOT frozen - entire model learns!
    """
    
    def __init__(self, esm_model_name="facebook/esm2_t33_650M_UR50D", hidden_dim=512, dropout=0.2):
        super().__init__()
        
        # Load pre-trained ESM-2 from HuggingFace
        print(f"Loading ESM-2 model: {esm_model_name}...")
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        
        # ✅ DO NOT FREEZE - Let ESM-2 learn!
        # All parameters are trainable by default
        
        esm_dim = self.esm_model.config.hidden_size  # 1280 for 650M, 2560 for 3B
        
        # Larger prediction head for better capacity
        self.predictor = nn.Sequential(
            nn.Linear(esm_dim * 2, hidden_dim),  # MHC + Peptide embeddings
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)  # IC50 prediction
        )
    
    def encode_sequence(self, sequences):
        """Encode sequences using ESM-2 and return mean-pooled embeddings"""
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(self.esm_model.parameters()).device) for k, v in inputs.items()}
        
        # ESM-2 forward pass (gradients enabled!)
        outputs = self.esm_model(**inputs)
        embeddings = outputs.last_hidden_state  # [batch, seq_len, esm_dim]
        
        # Mean pooling (exclude padding tokens)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        
        return embeddings
    
    def forward(self, mhc_sequences, peptide_sequences):
        # Encode sequences
        mhc_emb = self.encode_sequence(mhc_sequences)  # [batch, esm_dim]
        pep_emb = self.encode_sequence(peptide_sequences)  # [batch, esm_dim]
        
        # Concatenate and predict
        combined = torch.cat([mhc_emb, pep_emb], dim=1)  # [batch, esm_dim*2]
        predictions = self.predictor(combined)
        
        return predictions


# Copy the rest from esm2_simple.py (MHCPeptideDataset, train_epoch, evaluate, etc.)
# ... (same as before)
