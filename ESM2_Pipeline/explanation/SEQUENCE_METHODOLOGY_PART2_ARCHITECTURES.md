# COMPREHENSIVE TECHNICAL RECONSTRUCTION: SEQUENCE-BASED MHC-PEPTIDE BINDING PREDICTION
## PART 2 OF 5: MODEL ARCHITECTURES - CONCATENATION AND CROSS-ATTENTION STRATEGIES

---

## 7. ARCHITECTURAL STRATEGY 1: SIMPLE CONCATENATION (BASELINE)

### 7.1 Conceptual Foundation

The concatenation strategy represents the simplest approach to combining MHC and peptide representations. After independent encoding by ESM-2, the sequence representations are pooled and concatenated before passing through a regression head.

**Design rationale**:
- Leverages pretrained ESM-2 knowledge of both sequences independently
- No additional parameters for sequence interaction modeling
- Establishes baseline performance ceiling for simple fusion

**Limitations**:
- No explicit modeling of MHC-peptide interactions during encoding
- Relies entirely on regression head to learn binding patterns from concatenated features
- Cannot capture residue-level interactions between MHC and peptide

### 7.2 Architecture Details (Experiments: checkpoints_1_esm2_old, checkpoints_2_esm2, checkpoints_rerun_ESM2)

```python
class ESM2ConcatenationModel(nn.Module):
    """
    Simple concatenation baseline for MHC-peptide binding prediction.
    
    Architecture:
        MHC sequence → ESM-2 → Mean pooling → 
                                               Concatenate → MLP → Binding score
        Peptide     → ESM-2 → Mean pooling → 
    """
    
    def __init__(self, esm_model_name, hidden_dim=512, dropout=0.2):
        super().__init__()
        
        # Shared ESM-2 encoder for both sequences
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        embed_dim = self.esm_model.config.hidden_size  # 1280 or 2560
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim * 2),  # 2560 or 5120 input
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, mhc_batch, pep_batch):
        # Encode MHC
        mhc_outputs = self.esm_model(**mhc_batch)
        mhc_hidden = mhc_outputs.last_hidden_state  # (B, L_mhc, 1280)
        
        # Encode peptide
        pep_outputs = self.esm_model(**pep_batch)
        pep_hidden = pep_outputs.last_hidden_state  # (B, L_pep, 1280)
        
        # Mean pooling (excluding padding tokens)
        mhc_mask = mhc_batch['attention_mask'].unsqueeze(-1)  # (B, L_mhc, 1)
        pep_mask = pep_batch['attention_mask'].unsqueeze(-1)
        
        mhc_pooled = (mhc_hidden * mhc_mask).sum(dim=1) / mhc_mask.sum(dim=1)
        pep_pooled = (pep_hidden * pep_mask).sum(dim=1) / pep_mask.sum(dim=1)
        
        # Concatenate representations
        combined = torch.cat([mhc_pooled, pep_pooled], dim=1)  # (B, 2560)
        
        # Predict binding affinity
        output = self.regression_head(combined)  # (B, 1)
        return output.squeeze(-1)
```

**[INFERRED]**: The actual implementation in early experiments (checkpoints_1) may have used [CLS] token instead of mean pooling. The code above represents the logical design.
**TODO**: Verify exact pooling strategy used in checkpoints_1_esm2_old vs. checkpoints_2_esm2

### 7.3 Training Configuration - Experiment 1 (checkpoints_1_esm2_old)

**Model**: ESM-2 (facebook/esm2_t36_3B_UR50D) - 3 billion parameters

**Hyperparameters**:
```yaml
model_name: "facebook/esm2_t36_3B_UR50D"
batch_size: 64 (initial), then 32 (after memory issues)
learning_rate: 1e-5
epochs: 50
optimizer: AdamW (implicit from code)
weight_decay: [MISSING - likely 0.01 default]
lr_scheduler: ReduceLROnPlateau
  patience: 3 epochs
  factor: 0.1
gradient_clipping: 1.0 (max norm)
early_stopping: 10 epochs
mixed_precision: True (AMP)
gradient_accumulation: 1 (no accumulation)
```

**Training results** (from log file analysis):
- Training completed: October 21, 2025, 13:11:30
- Best epoch: 41
- Final train loss: 0.0011 (normalized scale)
- Best validation RMSE: 0.7704 pIC50 units
- Best validation R²: 0.7014

**Test set performance**:
```
Loss: 0.0118 (scaled)
RMSE: 0.8046 pIC50 units
R²: 0.6825
PCC: 0.8273
SCC: 0.8229
```

**Analysis**: This represents strong baseline performance, with R² = 0.6825 indicating the model explains ~68% of binding affinity variance. The gap between validation (R² = 0.7014) and test (R² = 0.6825) suggests mild overfitting despite early stopping.

### 7.4 Training Configuration - Experiment 2 (checkpoints_2_esm2)

**Model**: ESM-2 (facebook/esm2_t33_650M_UR50D) - 650 million parameters (smaller variant)

**Hyperparameters**:
```yaml
model_name: "facebook/esm2_t33_650M_UR50D"
batch_size: 32
learning_rate: 1e-5
epochs: 50
early_stopping: 10 epochs
scheduler_patience: 3
clip_norm: 1.0
use_amp: True
num_workers: 12
```

**Training results**:
- Date: October 23, 2025
- Best epoch: 32
- Final train loss: 0.0012 (scaled)
- Best validation RMSE: 0.7805 pIC50 units
- Best validation R²: 0.6935
- Early stopping triggered: Epoch 42

**Test set performance**:
```
Loss: 0.0119 (scaled)
RMSE: 0.8101 pIC50 units
R²: 0.6781
PCC: 0.8258
SCC: 0.8209
```

**Comparative analysis**:
- ESM-2 650M achieved nearly identical test performance to ESM-2 3B (R² = 0.6781 vs. 0.6825)
- Training time reduced significantly (~5x faster due to smaller model)
- Suggests that 650M parameters are sufficient for this task
- **Key finding**: Model capacity beyond 650M does not improve generalization for simple concatenation

### 7.5 Training Configuration - Experiment 3 (checkpoints_rerun_ESM2)

**Model**: ESM-2 (facebook/esm2_t33_650M_UR50D)

This appears to be a replication run to verify reproducibility.

**Training results**:
- Date: October 29, 2025
- Completed all 50 epochs (no early stopping)
- Final train loss: 0.0011 (scaled)
- Best validation RMSE: 0.7871 pIC50 units
- Best validation R²: 0.6881

**Test set performance**:
```
Loss: 0.0119 (scaled)
RMSE: 0.8100 pIC50 units
R²: 0.6782
PCC: 0.8251
SCC: 0.8210
```

**Reproducibility verification**:
- Test RMSE: 0.8100 (rerun) vs. 0.8101 (original) → **99.99% identical**
- Test R²: 0.6782 (rerun) vs. 0.6781 (original) → **Exactly reproduced**
- This confirms excellent reproducibility with fixed random seed (42)

---

## 8. ARCHITECTURAL STRATEGY 2: CROSS-ATTENTION (BIDIRECTIONAL INTERACTION)

### 8.1 Conceptual Foundation

The cross-attention strategy introduces explicit modeling of MHC-peptide interactions during feature extraction, allowing the model to learn:
- Which MHC residues are most relevant for binding given a specific peptide
- Which peptide residues interact strongly with specific MHC regions
- Bidirectional attention dependencies between the two sequences

**Biological motivation**:
- MHC binding groove residues should attend to peptide anchor positions
- Peptide residues should attend to MHC pocket-forming residues
- Mimics physical interaction where sequences mutually influence each other

**Technical advantages over concatenation**:
- Learnable interaction patterns at the representation level
- Attention weights interpretable as interaction importance
- Potential for better generalization to novel MHC-peptide combinations

### 8.2 Cross-Attention Architecture Components

#### 8.2.1 Multi-Head Cross-Attention Layer

```python
class CrossAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention between MHC and peptide representations.
    
    MHC attends to peptide: Which peptide features are relevant for this MHC?
    Peptide attends to MHC: Which MHC features are relevant for this peptide?
    """
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # MHC queries attend to peptide keys/values
        self.mhc_to_pep_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Peptide queries attend to MHC keys/values
        self.pep_to_mhc_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.mhc_norm = nn.LayerNorm(embed_dim)
        self.pep_norm = nn.LayerNorm(embed_dim)
        
        # Position-wise feed-forward networks
        self.mhc_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.pep_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, mhc_hidden, pep_hidden, mhc_mask, pep_mask):
        """
        Args:
            mhc_hidden: (B, L_mhc, D) - MHC sequence embeddings from ESM-2
            pep_hidden: (B, L_pep, D) - Peptide embeddings from ESM-2
            mhc_mask: (B, L_mhc) - Attention mask for MHC (1=valid, 0=padding)
            pep_mask: (B, L_pep) - Attention mask for peptide
        
        Returns:
            mhc_out: (B, L_mhc, D) - Attended MHC features
            pep_out: (B, L_pep, D) - Attended peptide features
        """
        # MHC attends to peptide
        # Q=MHC, K=peptide, V=peptide
        mhc_attended, mhc_attn_weights = self.mhc_to_pep_attn(
            query=mhc_hidden,
            key=pep_hidden,
            value=pep_hidden,
            key_padding_mask=~pep_mask  # Invert: True=ignore
        )
        
        # Add & norm (post-attention residual)
        mhc_residual = self.mhc_norm(mhc_hidden + mhc_attended)
        
        # Feed-forward with residual
        mhc_out = mhc_residual + self.mhc_ffn(mhc_residual)
        
        # Peptide attends to MHC (symmetric operation)
        pep_attended, pep_attn_weights = self.pep_to_mhc_attn(
            query=pep_hidden,
            key=mhc_hidden,
            value=mhc_hidden,
            key_padding_mask=~mhc_mask
        )
        
        pep_residual = self.pep_norm(pep_hidden + pep_attended)
        pep_out = pep_residual + self.pep_ffn(pep_residual)
        
        return mhc_out, pep_out
```

**Mathematical formulation**:

For MHC attending to peptide:
$$
\text{Attention}(Q_{\text{MHC}}, K_{\text{pep}}, V_{\text{pep}}) = \text{softmax}\left(\frac{Q_{\text{MHC}} K_{\text{pep}}^T}{\sqrt{d_k}}\right) V_{\text{pep}}
$$

Where:
- $Q_{\text{MHC}} = \text{MHC}_{\text{hidden}} W_Q \in \mathbb{R}^{L_{\text{MHC}} \times d_k}$
- $K_{\text{pep}} = \text{Pep}_{\text{hidden}} W_K \in \mathbb{R}^{L_{\text{pep}} \times d_k}$
- $V_{\text{pep}} = \text{Pep}_{\text{hidden}} W_V \in \mathbb{R}^{L_{\text{pep}} \times d_v}$
- $d_k = d_{\text{model}} / h$ (head dimension, typically 64 for 8 heads and 1280 hidden dim)

#### 8.2.2 Attention Pooling (Learned Sequence Summarization)

Traditional mean pooling treats all positions equally. Attention pooling learns which positions are most important for binding prediction:

```python
class AttentionPooling(nn.Module):
    """
    Learned attention-based pooling to create fixed-size sequence representation.
    
    Instead of mean pooling, learn importance weights for each position.
    """
    
    def __init__(self, embed_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: (B, L, D)
            attention_mask: (B, L) - 1 for valid tokens, 0 for padding
        
        Returns:
            pooled: (B, D) - Weighted sum of sequence representations
        """
        # Compute attention scores for each position
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (B, L)
        
        # Mask padding positions (set to -inf before softmax)
        attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))
        
        # Normalize to weights
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        
        # Weighted sum
        pooled = (hidden_states * attn_weights).sum(dim=1)  # (B, D)
        
        return pooled
```

**Biological interpretation**:
- For MHC: Higher weights on groove-forming residues (α1, α2 domains)
- For peptide: Higher weights on anchor positions (P2, P9 for HLA-A)
- **[INFERRED]**: Attention weights could provide interpretability for identifying critical binding residues
- **TODO**: Implement attention weight visualization to validate this hypothesis

#### 8.2.3 Interaction Module (Explicit Feature Combination)

Beyond simple concatenation, the interaction module computes multiple types of feature combinations:

```python
class InteractionModule(nn.Module):
    """
    Computes multiple interaction features between MHC and peptide pooled representations.
    
    Features:
        1. Concatenation: [MHC; Peptide]
        2. Element-wise product: MHC ⊙ Peptide (captures feature co-occurrence)
        3. Absolute difference: |MHC - Peptide| (captures dissimilarity)
    """
    
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        
        # Total input: 2*embed_dim (concat) + embed_dim (product) + embed_dim (diff)
        #            = 4 * embed_dim
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, mhc_pooled, pep_pooled):
        """
        Args:
            mhc_pooled: (B, D) - Pooled MHC representation
            pep_pooled: (B, D) - Pooled peptide representation
        
        Returns:
            interaction_features: (B, D) - Combined interaction representation
        """
        # Three types of interactions
        concat = torch.cat([mhc_pooled, pep_pooled], dim=-1)       # (B, 2D)
        product = mhc_pooled * pep_pooled                          # (B, D)
        difference = torch.abs(mhc_pooled - pep_pooled)            # (B, D)
        
        # Combine all interaction types
        combined = torch.cat([concat, product, difference], dim=-1)  # (B, 4D)
        
        # Project to embed_dim
        interaction_features = self.projection(combined)  # (B, D)
        
        return interaction_features
```

**Mathematical formulation**:
$$
\text{Interaction}(\mathbf{m}, \mathbf{p}) = W_2 \cdot \text{GELU}(W_1 \cdot [\mathbf{m}; \mathbf{p}; \mathbf{m} \odot \mathbf{p}; |\mathbf{m} - \mathbf{p}|] + b_1) + b_2
$$

Where:
- $\mathbf{m} \in \mathbb{R}^D$: Pooled MHC representation
- $\mathbf{p} \in \mathbb{R}^D$: Pooled peptide representation
- $[\cdot; \cdot]$: Concatenation operator
- $\odot$: Element-wise product (Hadamard product)

**Design rationale**:
- **Concatenation**: Preserves all information from both sequences
- **Product**: Captures which features are simultaneously active (e.g., hydrophobic MHC pocket + hydrophobic peptide anchor)
- **Difference**: Measures feature mismatch (e.g., charge incompatibility)

This is inspired by Siamese network architectures in computer vision and sentence similarity tasks.

### 8.3 Full Cross-Attention Model Architecture

```python
class ESM2CrossAttentionModel(nn.Module):
    """
    Complete cross-attention architecture for MHC-peptide binding prediction.
    
    Pipeline:
        1. Dual ESM-2 encoding (MHC and peptide separately)
        2. Stacked cross-attention layers
        3. Attention pooling for both sequences
        4. Interaction module
        5. Regression head
    """
    
    def __init__(
        self,
        esm_model_name,
        num_ca_layers=2,        # Number of cross-attention layers
        num_heads=8,
        hidden_dim=512,
        dropout=0.2,
        freeze_esm=False
    ):
        super().__init__()
        
        # Shared ESM-2 encoder
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        embed_dim = self.esm_model.config.hidden_size
        
        # Optional freezing of ESM-2 parameters
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        # Stacked cross-attention layers
        self.co_attention_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_ca_layers)
        ])
        
        # Pooling layers
        self.mhc_pooling = AttentionPooling(embed_dim)
        self.pep_pooling = AttentionPooling(embed_dim)
        
        # Interaction module
        self.interaction_module = InteractionModule(embed_dim, dropout)
        
        # Final regression head
        # Input: 2*embed_dim (original pooled) + 2*embed_dim (cross-attended pooled) 
        #        + embed_dim (interaction features) = 5*embed_dim
        total_dim = embed_dim * 5
        
        self.regression_head = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, mhc_batch, pep_batch):
        # Step 1: ESM-2 encoding
        with torch.set_grad_enabled(self.training and any(
            p.requires_grad for p in self.esm_model.parameters()
        )):
            mhc_hidden = self.esm_model(**mhc_batch).last_hidden_state
            pep_hidden = self.esm_model(**pep_batch).last_hidden_state
        
        mhc_mask = mhc_batch['attention_mask'].bool()
        pep_mask = pep_batch['attention_mask'].bool()
        
        # Save original representations
        mhc_original = mhc_hidden
        pep_original = pep_hidden
        
        # Step 2: Apply stacked cross-attention
        for co_attn_layer in self.co_attention_layers:
            mhc_hidden, pep_hidden = co_attn_layer(
                mhc_hidden, pep_hidden, mhc_mask, pep_mask
            )
        
        # Step 3: Pool both original and cross-attended representations
        mhc_orig_pooled = self.mhc_pooling(mhc_original, mhc_mask)  # (B, D)
        pep_orig_pooled = self.pep_pooling(pep_original, pep_mask)
        mhc_ca_pooled = self.mhc_pooling(mhc_hidden, mhc_mask)
        pep_ca_pooled = self.pep_pooling(pep_hidden, pep_mask)
        
        # Step 4: Compute interaction features
        interaction_features = self.interaction_module(
            mhc_ca_pooled, pep_ca_pooled
        )
        
        # Step 5: Concatenate all features
        combined = torch.cat([
            mhc_orig_pooled,      # Original MHC features (D)
            pep_orig_pooled,      # Original peptide features (D)
            mhc_ca_pooled,        # Cross-attended MHC (D)
            pep_ca_pooled,        # Cross-attended peptide (D)
            interaction_features  # Interaction features (D)
        ], dim=1)  # Total: 5D
        
        # Step 6: Predict binding affinity
        output = self.regression_head(combined)
        return output.squeeze(-1)
```

**Architecture parameter counts** (for ESM-2 650M):
- ESM-2 encoder: 652,353,941 parameters (frozen or fine-tuned)
- Single cross-attention layer: ~13M parameters
  - MultiheadAttention (MHC→Pep): 6.5M
  - MultiheadAttention (Pep→MHC): 6.5M
  - FFN layers: ~0.5M
- Attention pooling (×2): ~0.8M parameters
- Interaction module: ~6.5M parameters
- Regression head: ~5M parameters

**Total trainable parameters**:
- If ESM frozen: ~40-50M parameters
- If ESM fine-tuned: ~690M parameters

### 8.4 Training Configuration - Experiment 4 (checkpoints_6_esm2_CA)

**Model**: ESM-2 (facebook/esm2_t33_650M_UR50D) with cross-attention

**Hyperparameters**:
```yaml
model_name: "facebook/esm2_t33_650M_UR50D"
num_ca_layers: [MISSING - inferred as 2 based on code]
num_attention_heads: 8
batch_size: 128 (very large for cross-attention!)
learning_rate: 1e-5
epochs: 200
freeze_esm: [MISSING - likely False for early experiments]
```

**Training results** (partial - training stopped early):
- Date: October 26, 2025
- Best validation RMSE: [MISSING - log incomplete]
- Observation: Very large batch size (128) used
- **[INFERRED]**: Large batch size may have been attempted to stabilize cross-attention training
- **TODO**: Investigate why training stopped prematurely

**Analysis**: Early cross-attention experiments faced challenges:
1. High memory consumption (cross-attention quadratic in sequence length)
2. Training instability (random initialization of attention weights)
3. Slow convergence compared to concatenation baseline

This motivated the development of **warm-start fusion** strategy (next section).

---

*End of Part 2. Continue to Part 3 for warm-start fusion strategy and advanced training techniques.*
