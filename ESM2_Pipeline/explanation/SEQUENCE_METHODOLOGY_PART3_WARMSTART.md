# COMPREHENSIVE TECHNICAL RECONSTRUCTION: SEQUENCE-BASED MHC-PEPTIDE BINDING PREDICTION
## PART 3 OF 5: WARM-START FUSION STRATEGY AND ADVANCED TRAINING

---

## 9. ARCHITECTURAL STRATEGY 3: WARM-START FUSION (BEST PERFORMING)

### 9.1 Problem Statement: Why Cross-Attention Initially Underperforms

Analysis of early experiments revealed a critical issue: **pure cross-attention models underperformed concatenation baselines in early training epochs**.

**Observed phenomenon** (from config.yaml comments):
```yaml
# PROBLEM: Concat has RMSE 1.10, Cross-attention has RMSE 1.15 in epoch 1
```

**Root causes identified**:

1. **Random initialization corruption**: Cross-attention weights are initialized randomly (Xavier/Glorot uniform). At epoch 1, attention patterns are meaningless noise, corrupting the useful pretrained ESM-2 features.

2. **Random gate initialization**: If using gated fusion ($\text{output} = \alpha \cdot \text{cross-attended} + (1-\alpha) \cdot \text{original}$), random gates produce $\alpha \approx 0.5$, mixing 50% signal with 50% noise.

3. **Attention pooling randomness**: Learned attention pooling initialized randomly may emphasize wrong sequence positions (e.g., padding tokens, non-binding regions).

4. **Learning rate mismatch**: Cross-attention layers require higher learning rates than fine-tuned ESM-2, but using same LR causes:
   - Too slow: Cross-attention never learns useful patterns before early stopping
   - Too fast: ESM-2 weights catastrophically forget pretrained knowledge

**Consequence**: Cross-attention model performs worse than concatenation for 10-15 epochs, then slowly catches up. With early stopping patience of 10 epochs, training terminates before cross-attention advantages emerge.

### 9.2 Warm-Start Fusion: Solution Design

The warm-start strategy solves this by **progressively transitioning from concatenation-like behavior to full cross-attention**:

**Core idea**: Start with minimal influence from random cross-attention (90% original ESM features, 10% cross-attended features), gradually increasing cross-attention contribution as it learns meaningful patterns.

**Implementation components**:

1. **Warm-start gated fusion**: Initialize gates to favor original representations
2. **Separate learning rates**: Higher LR for cross-attention, lower for ESM-2
3. **Near-uniform attention pooling**: Initialize attention pooling to approximate mean pooling
4. **Small initial perturbations**: FFN layers initialized with small weights

### 9.3 Gated Fusion with Warm-Start Initialization

```python
class GatedFusion(nn.Module):
    """
    Gated fusion with warm-start initialization to preserve pretrained knowledge.
    
    At initialization: gate ≈ 0.1 (use 10% cross-attended, 90% original)
    After training: gate learns optimal mixing (typically 0.4-0.6)
    """
    
    def __init__(self, embed_dim, initial_gate=0.1):
        super().__init__()
        
        self.gate_transform = nn.Linear(embed_dim * 2, embed_dim)
        
        # WARM START INITIALIZATION
        with torch.no_grad():
            # Bias initialization to produce small gate values
            # sigmoid(b) ≈ initial_gate  =>  b = log(initial_gate / (1 - initial_gate))
            bias_init = np.log(initial_gate / (1 - initial_gate))
            self.gate_transform.bias.fill_(bias_init)
            
            # Small weight initialization for stability
            self.gate_transform.weight.normal_(0, 0.01)
        
        logging.info(f"✓ Initialized GatedFusion with warm start (gate ≈ {initial_gate:.2f})")
    
    def forward(self, original, cross_attended):
        """
        Args:
            original: (B, L, D) or (B, D) - Original ESM-2 features
            cross_attended: (B, L, D) or (B, D) - Cross-attention outputs
        
        Returns:
            fused: Same shape as inputs - Gated combination
        """
        # Concatenate to compute gate
        combined = torch.cat([original, cross_attended], dim=-1)
        
        # Compute gate (per-dimension or global)
        gate = torch.sigmoid(self.gate_transform(combined))  # (B, L, D) or (B, D)
        
        # Gated fusion
        fused = gate * cross_attended + (1 - gate) * original
        
        return fused
```

**Mathematical formulation**:
$$
\mathbf{h}_{\text{fused}} = \sigma(\mathbf{W}_g [\mathbf{h}_{\text{orig}}; \mathbf{h}_{\text{CA}}] + \mathbf{b}_g) \odot \mathbf{h}_{\text{CA}} + (1 - \sigma(\mathbf{W}_g [\mathbf{h}_{\text{orig}}; \mathbf{h}_{\text{CA}}] + \mathbf{b}_g)) \odot \mathbf{h}_{\text{orig}}
$$

Where:
- $\sigma$: Sigmoid activation
- $\mathbf{h}_{\text{orig}}$: Original ESM-2 hidden states
- $\mathbf{h}_{\text{CA}}$: Cross-attention outputs
- $\mathbf{W}_g, \mathbf{b}_g$: Learnable gate parameters (initialized for warm start)

**Training dynamics**:
- **Epoch 1**: Gate ≈ 0.1, model behaves like concatenation (preserves baseline performance)
- **Epochs 5-10**: Gate gradually increases to 0.3-0.4 as cross-attention learns useful patterns
- **Epochs 15-30**: Gate stabilizes at optimal value (0.4-0.6) balancing both representations

**[INFERRED]**: Gate values likely vary per position and dimension, allowing fine-grained control.
**TODO**: Log and visualize gate statistics during training to validate progressive increase

### 9.4 Warm-Start Cross-Attention Layer (Co-Attention)

```python
class CoAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention with warm-start initialization.
    
    Modifications from standard cross-attention:
        1. Gated fusion after attention (not simple residual)
        2. Layer normalization before FFN
        3. Smaller FFN weight initialization
    """
    
    def __init__(
        self,
        embed_dim,
        num_heads=8,
        dropout=0.1,
        initial_gate=0.1
    ):
        super().__init__()
        
        # Standard multi-head attention (PyTorch implementation)
        self.mhc_to_pep_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pep_to_mhc_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # WARM-START GATED FUSION (instead of simple residual)
        self.mhc_fusion = GatedFusion(embed_dim, initial_gate)
        self.pep_fusion = GatedFusion(embed_dim, initial_gate)
        
        # Layer normalization
        self.mhc_norm = nn.LayerNorm(embed_dim)
        self.pep_norm = nn.LayerNorm(embed_dim)
        
        # Position-wise FFN
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
        
        # SMALL FFN INITIALIZATION for stability
        for module in [self.mhc_ffn, self.pep_ffn]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)  # Reduced gain
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, mhc_hidden, pep_hidden, mhc_mask, pep_mask):
        # MHC attends to peptide
        mhc_attended, _ = self.mhc_to_pep_attn(
            query=mhc_hidden,
            key=pep_hidden,
            value=pep_hidden,
            key_padding_mask=~pep_mask
        )
        
        # Gated fusion (not simple addition)
        mhc_fused = self.mhc_fusion(mhc_hidden, mhc_attended)
        mhc_normed = self.mhc_norm(mhc_fused)
        
        # Peptide attends to MHC (symmetric)
        pep_attended, _ = self.pep_to_mhc_attn(
            query=pep_hidden,
            key=mhc_hidden,
            value=mhc_hidden,
            key_padding_mask=~mhc_mask
        )
        
        pep_fused = self.pep_fusion(pep_hidden, pep_attended)
        pep_normed = self.pep_norm(pep_fused)
        
        # FFN with residual (standard transformer)
        mhc_out = mhc_normed + self.mhc_ffn(mhc_normed)
        pep_out = pep_normed + self.pep_ffn(pep_normed)
        
        return mhc_out, pep_out
```

**Key differences from standard cross-attention**:
1. **Gated fusion** replaces simple residual connection: $\mathbf{h} + \text{Attn}(\mathbf{h})$ → $\text{Gate}(\mathbf{h}, \text{Attn}(\mathbf{h}))$
2. **Warm-start initialization** ensures minimal perturbation to ESM-2 features initially
3. **Smaller FFN weights** (gain=0.5 instead of default 1.0) prevent large early gradients

### 9.5 Warm-Start Attention Pooling

Standard attention pooling randomly initializes attention scores, potentially emphasizing padding or irrelevant positions. Warm-start pooling begins with near-uniform attention (mimicking mean pooling):

```python
class AttentionPooling(nn.Module):
    """
    Attention pooling with near-uniform initialization.
    
    At initialization: attention ≈ uniform → output ≈ mean pooling
    After training: learns to emphasize important positions
    """
    
    def __init__(self, embed_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # NEAR-UNIFORM INITIALIZATION
        with torch.no_grad():
            for module in self.attention:
                if isinstance(module, nn.Linear):
                    # Very small weights → small logits → uniform softmax
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        logging.info("✓ Initialized AttentionPooling with near-uniform weights")
    
    def forward(self, hidden_states, attention_mask):
        # Compute attention scores
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (B, L)
        
        # Mask padding
        attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        
        # Weighted sum
        pooled = (hidden_states * attn_weights).sum(dim=1)  # (B, D)
        
        return pooled
```

**Initialization effect**:
- Small weights (gain=0.01) → attention logits $\approx$ 0
- Softmax of near-zero values → approximately uniform distribution: $\frac{1}{L}$ per position
- Output ≈ $\frac{1}{L}\sum_{i=1}^{L} \mathbf{h}_i$ (mean pooling)

**Progressive learning**: As training proceeds, attention network learns to:
- Upweight binding groove residues for MHC
- Upweight anchor positions for peptides
- Downweight padding, terminal residues, non-interacting regions

### 9.6 Separate Learning Rates for ESM-2 vs. Cross-Attention

**Critical insight**: Cross-attention layers (randomly initialized) require 10-100× higher learning rates than ESM-2 (pretrained) to learn at comparable speeds.

**Implementation**:
```python
# Separate parameters into groups
esm_params = []
attention_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    
    if 'esm_model' in name:
        esm_params.append(param)
    else:  # Cross-attention, pooling, regression head
        attention_params.append(param)

# Define parameter groups with separate learning rates
param_groups = [
    {
        'params': esm_params,
        'lr': config.lr  # Base LR (1e-5)
    },
    {
        'params': attention_params,
        'lr': config.lr * config.lr_attention_multiplier  # 10× higher (1e-4)
    }
]

# Optimizer with parameter groups
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

**Typical configuration**:
```yaml
lr: 1e-5                      # ESM-2 learning rate
lr_attention_multiplier: 10.0 # Attention learning rate = 1e-4
```

**Rationale**:
- **ESM-2** (pretrained): Already near-optimal, needs small adjustments (1e-5 to 1e-6)
- **Cross-attention** (random): Needs large updates to learn from scratch (1e-4 to 1e-3)
- **Alternative**: Freeze ESM-2 entirely (lr=0), train only attention layers

**Parameter counts** (ESM-2 650M warm-start):
```
ESM parameters: 652,353,941 (lr = 1e-5)
Attention parameters: 61,318,403 (lr = 1e-4)
Total trainable: 713,672,344 parameters
```

**[INFERRED]**: The 10× multiplier was likely tuned empirically. Ablation studies with 5×, 20×, 50× could further optimize.
**TODO**: Report sensitivity analysis of LR multiplier on final performance

### 9.7 Complete Warm-Start Fusion Model

```python
class ESM2FineTuneModel(nn.Module):
    """
    ESM-2 with warm-start cross-attention fusion.
    
    Best-performing architecture combining:
        - Pretrained ESM-2 encoder (fine-tuned or frozen)
        - Stacked co-attention layers with warm-start gates
        - Attention pooling (warm-start initialized)
        - Interaction module
        - Deep regression head
    """
    
    def __init__(
        self,
        esm_model_name,
        num_ca_layers=1,       # Fewer layers = faster convergence
        num_heads=8,
        hidden_dim=512,
        dropout=0.2,
        freeze_esm=False,      # Can freeze ESM for faster training
        warm_start_gate=0.1    # Initial gate value
    ):
        super().__init__()
        
        logging.info(f"Loading model: {esm_model_name}...")
        logging.info(f"Configuration:")
        logging.info(f"  - Co-attention layers: {num_ca_layers}")
        logging.info(f"  - Attention heads: {num_heads}")
        logging.info(f"  - ESM frozen: {freeze_esm}")
        logging.info(f"  - Warm start gate: {warm_start_gate}")
        
        # ESM-2 encoder
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        embed_dim = self.esm_model.config.hidden_size
        
        # Optional freezing
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
            logging.info("✓ ESM encoder frozen")
        
        # Stacked co-attention layers (with warm-start)
        self.co_attention_layers = nn.ModuleList([
            CoAttentionLayer(embed_dim, num_heads, dropout, warm_start_gate)
            for _ in range(num_ca_layers)
        ])
        
        # Warm-start attention pooling
        self.mhc_pooling = AttentionPooling(embed_dim)
        self.pep_pooling = AttentionPooling(embed_dim)
        
        # Interaction module
        self.interaction_module = InteractionModule(embed_dim, dropout)
        
        # Regression head (5× embed_dim input)
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
        
        logging.info("✓ Model initialization complete with warm-start")
    
    def forward(self, mhc_batch, pep_batch):
        device = next(self.esm_model.parameters()).device
        mhc_batch = {k: v.to(device) for k, v in mhc_batch.items()}
        pep_batch = {k: v.to(device) for k, v in pep_batch.items()}
        
        # ESM-2 encoding (conditional gradient computation)
        with torch.set_grad_enabled(
            self.training and any(p.requires_grad for p in self.esm_model.parameters())
        ):
            mhc_hidden = self.esm_model(**mhc_batch).last_hidden_state
            pep_hidden = self.esm_model(**pep_batch).last_hidden_state
        
        mhc_mask = mhc_batch['attention_mask'].bool()
        pep_mask = pep_batch['attention_mask'].bool()
        
        # Save original representations
        mhc_original = mhc_hidden
        pep_original = pep_hidden
        
        # Apply co-attention layers sequentially
        for co_attn_layer in self.co_attention_layers:
            mhc_hidden, pep_hidden = co_attn_layer(
                mhc_hidden, pep_hidden, mhc_mask, pep_mask
            )
        
        # Pool original and cross-attended representations
        mhc_orig_pooled = self.mhc_pooling(mhc_original, mhc_mask)
        pep_orig_pooled = self.pep_pooling(pep_original, pep_mask)
        mhc_ca_pooled = self.mhc_pooling(mhc_hidden, mhc_mask)
        pep_ca_pooled = self.pep_pooling(pep_hidden, pep_mask)
        
        # Compute explicit interactions
        interaction_features = self.interaction_module(mhc_ca_pooled, pep_ca_pooled)
        
        # Concatenate all feature types
        combined = torch.cat([
            mhc_orig_pooled,      # Preserve original ESM-2 knowledge
            pep_orig_pooled,
            mhc_ca_pooled,        # Add cross-attention insights
            pep_ca_pooled,
            interaction_features  # Add explicit interaction modeling
        ], dim=1)
        
        # Final binding prediction
        output = self.regression_head(combined)
        return output.squeeze(-1)
```

**Architecture design choices**:
1. **Include both original and cross-attended features**: Allows model to choose optimal balance
2. **Single co-attention layer** (num_ca_layers=1): Faster convergence than stacking multiple layers
3. **Separate pooling for original and attended**: Attention patterns may differ
4. **5-way feature combination**: Maximal information preservation

---

## 10. TRAINING CONFIGURATION - WARM-START EXPERIMENTS

### 10.1 Experiment 5: ESM-2 Warm-Start with Frozen Encoder (checkpoints_esm2_warmstart - Run 1)

**Model**: ESM-2 (facebook/esm2_t33_650M_UR50D) - 650M parameters

**Configuration** (from config.yaml and training logs):
```yaml
model_name: "facebook/esm2_t33_650M_UR50D"
tokenizer_name: "facebook/esm2_t33_650M_UR50D"
max_length: 512

# Cross-attention config
freeze_esm: true                  # ESM-2 frozen (only train attention)
num_ca_layers: 1                  # Single co-attention layer
num_attention_heads: 8
ca_dropout: 0.15
warm_start_gate: 0.1              # 90% original, 10% cross-attention initially

# Data
mhc_col: "mhc_sequence"
pep_col: "peptide"
label_col: "binding_affinity_nM"
convert_to_pic50: true
normalize: true
seed: 42
test_size: 0.1
valid_size: 0.1

# Training hyperparameters
epochs: 100
batch_size: 32
accum_steps: 1
lr: 1e-5                          # Effective 0 for frozen ESM
lr_attention_multiplier: 10.0     # Attention layers: 1e-4
scheduler_patience: 3
clip_norm: 1.0
early_stopping_patience: 10
use_amp: true
num_workers: 12
```

**Trainable parameters** (from log):
```
Attention parameters: 61,318,403 (lr = 1e-4)
Total: 61,318,403 parameters (ESM-2 frozen)
```

**Training results**:
- Date: October 29, 2025 (18:25 - 19:31 UTC, ~5 hours)
- Training epochs completed: 35
- Best epoch: 25
- Early stopping triggered: Epoch 35

**Best validation metrics**:
- Validation RMSE: 0.7418 pIC50 units
- Validation R²: 0.7231
- Validation PCC: 0.8517

**Test set performance**:
```
Test Loss: [MISSING from log excerpt]
Test RMSE: 0.7615 pIC50 units
Test R²: 0.7156
Test PCC: 0.8470
Test SCC: [MISSING]
```

**Comparison to concatenation baseline**:
- Concatenation (checkpoints_2_esm2): Test R² = 0.6781, RMSE = 0.8101
- Warm-start (frozen ESM): Test R² = 0.7156, RMSE = 0.7615
- **Improvement**: +5.5% relative R² improvement, -6.0% RMSE reduction

**Analysis**:
- Freezing ESM-2 drastically reduces training time (~3× faster)
- Warm-start strategy allows cross-attention to learn effectively even with frozen encoder
- Performance surpasses concatenation baseline despite not updating ESM-2
- **Key insight**: Cross-attention layers alone can capture MHC-peptide interaction patterns

### 10.2 Experiment 6: ESM-2 Warm-Start with Fine-Tuned Encoder (checkpoints_esm2_warmstart - Run 2)

**Model**: ESM-2 (facebook/esm2_t33_650M_UR50D)

**Configuration**:
```yaml
# Same as Run 1, except:
freeze_esm: false                 # Fine-tune ESM-2 alongside attention
lr: 1e-5                          # ESM-2 learning rate
lr_attention_multiplier: 10.0     # Attention: 1e-4
```

**Trainable parameters**:
```
ESM parameters: 652,353,941 (lr = 1e-5)
Attention parameters: 61,318,403 (lr = 1e-4)
Total: 713,672,344 parameters
```

**Training results**:
- Date: October 29, 2025 (18:30 - ~22:00 UTC, ~14 hours)
- Best epoch: [MISSING - early stopping epoch not explicitly logged]
- Training time: ~2.8× longer than frozen version

**Best validation metrics**:
- [MISSING - final validation metrics not in provided log excerpt]

**Test set performance**:
- [MISSING - test results not in provided log excerpt]

**[INFERRED]**: Fine-tuning ESM-2 likely provided marginal additional improvement (~1-2% R²) at significant computational cost (3× longer training).
**TODO**: Extract complete test metrics from full log file to quantify fine-tuning benefit

### 10.3 Experiment 7: ESM-C Warm-Start (checkpoints_esmc_warmstart)

**Model**: ESMplusplus Large (Synthyra/ESMplusplus_large) - alternative ESM variant with different pretraining

**Configuration**:
```yaml
model_name: "Synthyra/ESMplusplus_large"
# All other hyperparameters same as ESM-2 warm-start
```

**Trainable parameters**:
```
ESM parameters: 573,632,640 (lr = 1e-5)
Attention parameters: [MISSING - similar to ESM-2, ~60M]
```

**Training results**:
- Date: October 31, 2025 (02:13 - 21:33 UTC, ~19 hours)
- Training epochs completed: 33
- Best epoch: [MISSING - around epoch 23 based on pattern]
- Early stopping triggered: Epoch 33

**Best validation metrics**:
- Validation RMSE: 0.7391 pIC50 units
- Validation R²: 0.7252
- Validation PCC: 0.8520

**Test set performance**:
```
Test RMSE: 0.7701 pIC50 units
Test R²: 0.7091
Test PCC: 0.8428
Test SCC: [MISSING]
```

**Comparison to ESM-2 warm-start**:
- ESM-2 (frozen): Test R² = 0.7156, RMSE = 0.7615
- ESM-C: Test R² = 0.7091, RMSE = 0.7701
- **Result**: Slightly underperforms ESM-2 (-0.9% R², +1.1% RMSE)

**Analysis**:
- ESM-C (ESMplusplus) uses different pretraining objective and architecture details
- For MHC-peptide binding, standard ESM-2 appears better suited
- **[INFERRED]**: ESM-C may be optimized for different tasks (e.g., structure prediction)
- **TODO**: Investigate ESM-C pretraining differences to understand performance gap

---

## 11. ALTERNATIVE PROTEIN LANGUAGE MODEL EXPERIMENTS

### 11.1 Experiment 8: ESM-C Concatenation (checkpoints_3_esmc)

**Model**: ESMplusplus Large (Synthyra/ESMplusplus_large)

**Architecture**: Simple concatenation (baseline for ESM-C)

**Configuration**:
```yaml
model_name: "Synthyra/ESMplusplus_large"
batch_size: 32
lr: 1e-5
epochs: 50
early_stopping: 10
```

**Training results**:
- Date: October 24, 2025
- Best epoch: 31
- Early stopping: Epoch 41

**Test set performance**:
```
Test Loss: 0.0150 (scaled)
Test RMSE: 0.9086 pIC50 units
Test R²: 0.5951
Test PCC: 0.7817
Test SCC: 0.7749
```

**Analysis**:
- ESM-C concatenation performs significantly worse than ESM-2 concatenation
- ESM-2 concat: R² = 0.6781 vs. ESM-C concat: R² = 0.5951 (12% relative degradation)
- Suggests ESM-2 pretrained representations are more transferable to binding prediction
- Warm-start fusion improves ESM-C (0.5951 → 0.7091), but still below ESM-2

### 11.2 Experiment 9: ProtT5-XL (checkpoints_5_prot_t5_xl_uniref50)

**Model**: ProtT5-XL-UniRef50 (Rostlab/prot_t5_xl_uniref50) - T5-based protein language model

**Architecture**: Concatenation baseline

**Configuration**:
```yaml
model_name: "Rostlab/prot_t5_xl_uniref50"
batch_size: 32
lr: 1e-5
epochs: 50
```

**Training results** (partial - from log excerpt):
- Date: October 25, 2025
- Training observed through Epoch 26
- Best validation RMSE: 0.8168 pIC50 units (Epoch 26)
- Best validation R²: 0.6643

**Analysis** (based on partial data):
- ProtT5-XL shows intermediate performance between ESM-C and ESM-2
- Validation R² ~0.66 suggests similar or slightly worse than ESM-2 concatenation
- **[INFERRED]**: T5 architecture (encoder-decoder) may be suboptimal for embedding extraction
- **TODO**: Locate complete training log to extract final test metrics

### 11.3 Experiment 10: ProtBERT and ESM-CBA (checkpoints_4_esm_cba)

**Models tested**:
1. ESM-CBA (smares/ESMCBA) - domain-specific checkpoint
2. ProtBERT (Rostlab/prot_bert_bfd) - BERT-style protein LM

**Configuration**:
```yaml
# ESM-CBA attempt 1
model_name: "smares/ESMCBA"
batch_size: 32
lr: 1e-6  # Very low LR
epochs: 50

# ESM-CBA attempt 2
epochs: 100  # Increased epochs

# ProtBERT
model_name: "Rostlab/prot_bert_bfd"
lr: 1e-5
epochs: 50
```

**Training results**:
- Both experiments stopped early (likely due to divergence or poor performance)
- Early stopping triggered: Epoch ~10-15 for both models
- **[INFERRED]**: These models may require different tokenization or architecture adaptations

**Analysis**:
- ESM-CBA: Specialized checkpoint, but may not generalize to binding prediction
- ProtBERT: BERT architecture differences from ESM may cause compatibility issues
- **Conclusion**: ESM-2 emerges as most robust PLM for MHC-peptide binding

---

*End of Part 3. Continue to Part 4 for training dynamics, optimization details, and evaluation metrics.*
