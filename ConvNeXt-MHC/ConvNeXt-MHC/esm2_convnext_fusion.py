"""
ESM-2 + ConvNeXt Fusion Architecture for MHC-Peptide Binding Prediction

This module integrates Meta's ESM-2 (Evolutionary Scale Modeling) protein language model
with ConvNeXt's degenerate encoding for multi-modal MHC-peptide binding prediction.

ESM-2 advantages:
- Captures evolutionary patterns learned from 65M+ protein sequences
- Provides rich per-residue embeddings (320-5120 dim depending on model)
- Understands protein structure and function from sequence alone
- State-of-the-art on protein structure prediction tasks

Architecture Options:
1. ESM-2 only (replace degenerate encoding)
2. ESM-2 + ConvNeXt parallel fusion (best performance)
3. ESM-2 → ConvNeXt sequential (use ESM embeddings as input)

Requirements:
    pip install fair-esm
    pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import esm

# Try TensorFlow imports (for ConvNeXt)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Only PyTorch models will work.")


class ESM2Encoder(nn.Module):
    """
    ESM-2 encoder for extracting protein embeddings
    
    Available ESM-2 models:
    - esm2_t6_8M_UR50D: 6 layers, 8M params, 320 dim (fastest)
    - esm2_t12_35M_UR50D: 12 layers, 35M params, 480 dim
    - esm2_t30_150M_UR50D: 30 layers, 150M params, 640 dim
    - esm2_t33_650M_UR50D: 33 layers, 650M params, 1280 dim (recommended)
    - esm2_t36_3B_UR50D: 36 layers, 3B params, 2560 dim (best, slow)
    - esm2_t48_15B_UR50D: 48 layers, 15B params, 5120 dim (largest)
    """
    
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        freeze_esm: bool = False,
        pool_method: str = "mean"
    ):
        """
        Args:
            model_name: ESM-2 model variant to use
            freeze_esm: If True, don't fine-tune ESM weights
            pool_method: How to pool sequence embeddings ('mean', 'max', 'cls', 'attention')
        """
        super().__init__()
        
        # Load pre-trained ESM-2 model
        self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Freeze ESM weights if requested
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        self.pool_method = pool_method
        self.embed_dim = self.esm_model.embed_dim
        
        # Attention pooling layer (if using attention pooling)
        if pool_method == "attention":
            self.attention_weights = nn.Linear(self.embed_dim, 1)
    
    def forward(
        self,
        sequences: List[str],
        return_per_residue: bool = False
    ) -> torch.Tensor:
        """
        Extract ESM-2 embeddings for a batch of sequences
        
        Args:
            sequences: List of protein sequences (e.g., ["AILGFVFTL", "GILGFVFTL"])
            return_per_residue: If True, return per-residue embeddings
        
        Returns:
            embeddings: [batch_size, embed_dim] or [batch_size, seq_len, embed_dim]
        """
        # Convert sequences to ESM format
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        
        # Move to same device as model
        batch_tokens = batch_tokens.to(next(self.esm_model.parameters()).device)
        
        # Extract ESM-2 representations
        with torch.no_grad() if not self.training else torch.enable_grad():
            results = self.esm_model(batch_tokens, repr_layers=[self.esm_model.num_layers])
            embeddings = results["representations"][self.esm_model.num_layers]
        
        # Remove <cls> and <eos> tokens (first and last)
        embeddings = embeddings[:, 1:-1, :]  # [batch, seq_len, embed_dim]
        
        if return_per_residue:
            return embeddings
        
        # Pool to fixed-size representation
        if self.pool_method == "mean":
            pooled = embeddings.mean(dim=1)
        elif self.pool_method == "max":
            pooled = embeddings.max(dim=1)[0]
        elif self.pool_method == "cls":
            # Use <cls> token representation
            pooled = results["representations"][self.esm_model.num_layers][:, 0, :]
        elif self.pool_method == "attention":
            # Attention-weighted pooling
            attention_scores = self.attention_weights(embeddings)  # [batch, seq_len, 1]
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled = (embeddings * attention_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        return pooled


class ESM2_ConvNeXt_Fusion(nn.Module):
    """
    Multi-modal fusion of ESM-2 embeddings and ConvNeXt degenerate encoding
    
    Architecture:
        MHC Sequence ──→ ESM-2 ──→ [batch, 1280]  ──┐
                                                      │
                                                      ├──→ Cross-Attention ──→ Prediction
                                                      │
        Peptide     ──→ ESM-2 ──→ [batch, 1280]  ──┘
        
        OR (if using ConvNeXt):
        
        Degenerate  ──→ ConvNeXt ──→ [batch, 128] ──┐
        Encoding                                      │
                                                      ├──→ Cross-Attention ──→ Prediction
                                                      │
        Peptide     ──→ ESM-2    ──→ [batch, 1280] ──┘
    """
    
    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        use_convnext: bool = True,
        freeze_esm: bool = True
    ):
        super().__init__()
        
        # ESM-2 encoders for MHC and peptide
        self.mhc_encoder = ESM2Encoder(esm_model_name, freeze_esm=freeze_esm, pool_method="mean")
        self.peptide_encoder = ESM2Encoder(esm_model_name, freeze_esm=freeze_esm, pool_method="mean")
        
        esm_dim = self.mhc_encoder.embed_dim
        
        self.use_convnext = use_convnext
        
        # Project ESM embeddings to hidden_dim
        self.mhc_projection = nn.Linear(esm_dim, hidden_dim)
        self.peptide_projection = nn.Linear(esm_dim, hidden_dim)
        
        # Optional: ConvNeXt branch (if using degenerate encoding)
        if use_convnext:
            # This would be a PyTorch version of ConvNeXt
            # For now, we'll add a placeholder CNN
            self.convnext_encoder = nn.Sequential(
                nn.Conv2d(21, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.convnext_projection = nn.Linear(128, hidden_dim)
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        fusion_input_dim = hidden_dim * 2  # MHC + Peptide
        if use_convnext:
            fusion_input_dim += hidden_dim  # + ConvNeXt
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Regression output
        )
    
    def forward(
        self,
        mhc_sequences: List[str],
        peptide_sequences: List[str],
        degenerate_encoding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            mhc_sequences: List of MHC sequences
            peptide_sequences: List of peptide sequences
            degenerate_encoding: Optional [batch, 20, 9, 21] degenerate encoding
        
        Returns:
            predictions: [batch, 1] pIC50 predictions
        """
        # Extract ESM-2 embeddings
        mhc_embeddings = self.mhc_encoder(mhc_sequences)  # [batch, esm_dim]
        peptide_embeddings = self.peptide_encoder(peptide_sequences)  # [batch, esm_dim]
        
        # Project to hidden_dim
        mhc_features = self.mhc_projection(mhc_embeddings)  # [batch, hidden_dim]
        peptide_features = self.peptide_projection(peptide_embeddings)  # [batch, hidden_dim]
        
        # Prepare for cross-attention (add sequence dimension)
        mhc_features_seq = mhc_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        peptide_features_seq = peptide_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention: MHC attends to Peptide
        attended_features, _ = self.cross_attention(
            query=mhc_features_seq,
            key=peptide_features_seq,
            value=peptide_features_seq
        )
        attended_features = attended_features.squeeze(1)  # [batch, hidden_dim]
        
        # Combine features
        combined_features = [mhc_features, peptide_features]
        
        # Add ConvNeXt features if available
        if self.use_convnext and degenerate_encoding is not None:
            # Degenerate encoding: [batch, 20, 9, 21]
            # Permute to [batch, 21, 20, 9] for Conv2d
            degenerate_encoding = degenerate_encoding.permute(0, 3, 1, 2)
            convnext_features = self.convnext_encoder(degenerate_encoding)  # [batch, 128]
            convnext_features = self.convnext_projection(convnext_features)  # [batch, hidden_dim]
            combined_features.append(convnext_features)
        
        # Concatenate all features
        fused_features = torch.cat(combined_features, dim=1)
        
        # Final prediction
        predictions = self.fusion(fused_features)
        
        return predictions


class ESM2_Only_Model(nn.Module):
    """
    ESM-2 only model (no ConvNeXt, pure language model approach)
    
    Simpler architecture that uses only ESM-2 embeddings for both MHC and peptide.
    This is faster and requires less memory than the fusion model.
    """
    
    def __init__(
        self,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_esm: bool = True
    ):
        super().__init__()
        
        self.mhc_encoder = ESM2Encoder(esm_model_name, freeze_esm=freeze_esm)
        self.peptide_encoder = ESM2Encoder(esm_model_name, freeze_esm=freeze_esm)
        
        esm_dim = self.mhc_encoder.embed_dim
        
        # Simple MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(esm_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        mhc_sequences: List[str],
        peptide_sequences: List[str]
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            mhc_sequences: List of MHC sequences
            peptide_sequences: List of peptide sequences
        
        Returns:
            predictions: [batch, 1] pIC50 predictions
        """
        mhc_embeddings = self.mhc_encoder(mhc_sequences)
        peptide_embeddings = self.peptide_encoder(peptide_sequences)
        
        # Concatenate and predict
        combined = torch.cat([mhc_embeddings, peptide_embeddings], dim=1)
        predictions = self.predictor(combined)
        
        return predictions


def create_esm2_convnext_model(
    architecture: str = "fusion",
    esm_model: str = "esm2_t33_650M_UR50D",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """
    Factory function to create ESM-2 + ConvNeXt models
    
    Args:
        architecture: 'esm_only', 'fusion', or 'esm_to_convnext'
        esm_model: ESM-2 model variant
        device: 'cuda' or 'cpu'
    
    Returns:
        model: PyTorch model
    """
    if architecture == "esm_only":
        model = ESM2_Only_Model(
            esm_model_name=esm_model,
            hidden_dim=256,
            dropout=0.1,
            freeze_esm=True
        )
    elif architecture == "fusion":
        model = ESM2_ConvNeXt_Fusion(
            esm_model_name=esm_model,
            hidden_dim=256,
            num_attention_heads=8,
            dropout=0.1,
            use_convnext=True,
            freeze_esm=True
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model.to(device)


# ============================================================================
# Training utilities
# ============================================================================

def train_esm2_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda"
):
    """
    Training loop for ESM-2 models
    
    Args:
        model: ESM2_Only_Model or ESM2_ConvNeXt_Fusion
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            mhc_seqs = batch['mhc_sequences']
            pep_seqs = batch['peptide_sequences']
            labels = batch['pic50'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, ESM2_ConvNeXt_Fusion):
                degenerate = batch.get('degenerate_encoding', None)
                if degenerate is not None:
                    degenerate = degenerate.to(device)
                predictions = model(mhc_seqs, pep_seqs, degenerate)
            else:
                predictions = model(mhc_seqs, pep_seqs)
            
            loss = criterion(predictions.squeeze(), labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                mhc_seqs = batch['mhc_sequences']
                pep_seqs = batch['peptide_sequences']
                labels = batch['pic50'].to(device)
                
                if isinstance(model, ESM2_ConvNeXt_Fusion):
                    degenerate = batch.get('degenerate_encoding', None)
                    if degenerate is not None:
                        degenerate = degenerate.to(device)
                    predictions = model(mhc_seqs, pep_seqs, degenerate)
                else:
                    predictions = model(mhc_seqs, pep_seqs)
                
                loss = criterion(predictions.squeeze(), labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_esm2_model.pt')
            print(f"  → Saved new best model (val_loss={val_loss:.4f})")


if __name__ == "__main__":
    print("ESM-2 + ConvNeXt Integration Module")
    print("=" * 70)
    print("\nAvailable ESM-2 models:")
    print("  - esm2_t6_8M_UR50D      (8M params, 320 dim)   ← Fast, testing")
    print("  - esm2_t12_35M_UR50D    (35M params, 480 dim)")
    print("  - esm2_t30_150M_UR50D   (150M params, 640 dim)")
    print("  - esm2_t33_650M_UR50D   (650M params, 1280 dim) ← Recommended")
    print("  - esm2_t36_3B_UR50D     (3B params, 2560 dim)")
    print("  - esm2_t48_15B_UR50D    (15B params, 5120 dim)  ← Best (slow)")
    
    print("\nArchitecture options:")
    print("  1. ESM-2 only          - Fast, pure language model")
    print("  2. ESM-2 + ConvNeXt    - Multi-modal fusion (best performance)")
    
    print("\nExample usage:")
    print("  model = create_esm2_convnext_model(architecture='esm_only')")
    print("  predictions = model(['MHCSEQ'], ['PEPTIDE'])")
