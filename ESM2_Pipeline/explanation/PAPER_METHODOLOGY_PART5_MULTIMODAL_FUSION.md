# COMPREHENSIVE MULTIMODAL MHC-PEPTIDE BINDING PREDICTION METHODOLOGY

## PART 5/8: MULTIMODAL FUSION ARCHITECTURES

---

## 7. Integrating Sequence and Structure Modalities

### 7.1 Motivation for Multimodal Learning

**Central Hypothesis:** Sequence and structure representations capture **complementary information** about MHC-peptide binding:

**Sequence (ESM-2) Strengths:**
- Evolutionary conservation patterns across homologous proteins
- Functional motifs learned from 65M UniRef sequences  
- Implicit structural knowledge encoded in pretrained weights
- Strong performance on high-data alleles (R² = 0.7156)

**Sequence Limitations:**
- Cannot distinguish geometric isomers (same sequence, different structure)
- Limited on rare alleles with few evolutionary relatives
- Misses explicit spatial contacts and binding geometry

**Structure (EGNN) Strengths:**
- Direct encoding of 3D binding interface geometry
- Spatial proximity relationships (contact networks within 4-10 Å)
- Transferable across species (structural conservation)
- Robust to sequence divergence

**Structure Limitations:**
- Dependent on AlphaFold2 prediction quality (pLDDT variations)
- Limited coverage (58-68% of dataset with reliable structures)
- Lacks evolutionary context from protein families
- Underperforms sequence-only models (expected R² = 0.65-0.71)

**Fusion Objective:** Combine modalities to achieve **synergistic improvement**:
$$
\text{Performance}_{\text{multimodal}} > \max(\text{Performance}_{\text{sequence}}, \text{Performance}_{\text{structure}})
$$

**Target Performance:** R² = 0.74-0.76 (5-10% improvement over ESM-2 650M warm-start baseline of 0.7156)

---

### 7.2 Multimodal Fusion Strategies: Taxonomy

**Fusion Level Taxonomy:**

#### 7.2.1 Early Fusion (Feature Concatenation)

**Approach:** Combine raw sequence and structure features before model processing.

**Architecture:**
```python
# Concatenate sequence tokens with structure coordinates
combined_input = torch.cat([sequence_tokens, structure_coords], dim=-1)
fused_model = TransformerEncoder(combined_input)
```

**Advantages:**
- Simple implementation
- Maximum information sharing across modalities
- Models can learn cross-modal interactions from scratch

**Disadvantages:**
- Loses pretrained knowledge (cannot use frozen ESM-2)
- Requires retraining from scratch (computationally expensive)
-難 to handle missing modalities (structure unavailable for 30-40% of samples)

**Verdict:** Not suitable for our use case (sacrifices ESM-2 pretraining).

#### 7.2.2 Late Fusion (Decision-Level)

**Approach:** Train separate models for sequence and structure, then combine predictions.

**Architecture:**
```python
# Train independently
seq_pred = ESM2Model(sequence)
struct_pred = EGNNModel(structure)

# Combine predictions
final_pred = weighted_average(seq_pred, struct_pred, weights=[0.7, 0.3])
# OR
final_pred = MetaLearner([seq_pred, struct_pred])
```

**Advantages:**
- Leverages pretrained models (ESM-2, EGNN)
- Handles missing modalities gracefully (fall back to sequence-only)
- Simple ensemble approach

**Disadvantages:**
- No interaction between modalities during learning
- Suboptimal information integration (decisions already made)
- Requires careful weight tuning

**Verdict:** Solid baseline but limited by lack of cross-modal learning.

#### 7.2.3 Intermediate Fusion (Feature-Level)

**Approach:** Extract embeddings from separate models, then fuse before final prediction.

**Architecture (Our Approach):**
```python
# Extract embeddings
seq_embedding = ESM2Encoder(sequence)  # (batch, 1280)
struct_embedding = EGNNEncoder(structure)  # (batch, 128)

# Fuse embeddings
fused = FusionModule(seq_embedding, struct_embedding)  # (batch, fusion_dim)

# Predict binding affinity
prediction = RegressionHead(fused)
```

**Advantages:**
- Preserves pretrained representations (ESM-2, EGNN)
- Allows cross-modal attention and gating
- Flexible fusion mechanisms (concatenation, attention, bilinear)
- Handles missing modalities with learned fallback strategies

**Disadvantages:**
- Requires careful architecture design
- Potential for embedding dimension mismatch (1280 vs. 128)
- Training complexity (three components: sequence encoder, structure encoder, fusion)

**Verdict:** **Optimal choice** for our task—balances pretraining preservation and cross-modal learning.

---

### 7.3 Proposed Multimodal Architecture

**Overall Pipeline:**

```
Input: (MHC_seq, Peptide_seq, MHC_structure, Peptide_structure)
  ↓
[Sequence Branch]
  ESM-2 650M Warm-Start (frozen or fine-tuned)
  → MHC_seq_emb (1280-dim), Peptide_seq_emb (1280-dim)
  → Cross-Attention Fusion
  → Seq_embedding (1280-dim)
  ↓
[Structure Branch]
  EGNN (4 layers, trained from scratch)
  → MHC_struct_emb (128-dim), Peptide_struct_emb (128-dim)
  → Graph Pooling
  → Struct_embedding (128-dim)
  ↓
[Multimodal Fusion]
  Gated Fusion + Cross-Modal Attention
  → Fused_embedding (512-dim)
  ↓
[Regression Head]
  3-layer MLP
  → Binding_affinity (pIC50)
```

---

### 7.4 Sequence Encoder (Reusing ESM-2 Warm-Start)

**Rationale:** Our best sequence-only model (ESM-2 650M warm-start, R² = 0.7156) already encodes rich binding information. **Reuse this trained model** as the sequence encoder to avoid retraining.

**Configuration:**

```python
class SequenceEncoder(nn.Module):
    """
    Reuse ESM-2 warm-start cross-attention model
    Optionally freeze for faster training
    """
    def __init__(self, checkpoint_path: str, freeze: bool = True):
        super().__init__()
        
        # Load pretrained ESM-2 warm-start model
        self.model = ESM2FineTuneModel.load_from_checkpoint(checkpoint_path)
        
        # Freeze weights if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Sequence encoder frozen (using pretrained)")
        
        # Embedding dimension
        self.output_dim = 1280  # ESM-2 650M embedding size
    
    def forward(self, mhc_tokens, peptide_tokens):
        """
        Returns:
            seq_embedding: (batch_size, 1280) pooled sequence representation
        """
        # Run ESM-2 warm-start model
        # (includes cross-attention between MHC and peptide)
        with torch.set_grad_enabled(self.training and not self.freeze):
            seq_embedding = self.model.get_embedding(mhc_tokens, peptide_tokens)
        
        return seq_embedding  # (batch, 1280)
```

**Freezing Strategy:**

**Option 1: Freeze Sequence Encoder**
- **Advantages**: Faster training (only train structure encoder + fusion), prevents overfitting
- **Disadvantages**: Sequence encoder cannot adapt to structure information
- **When to use**: Limited GPU memory, want fast experimentation

**Option 2: Fine-Tune Sequence Encoder**
- **Advantages**: Sequence encoder learns structure-aware representations
- **Disadvantages**: Slower training, risk of catastrophic forgetting
- **When to use**: Sufficient compute, final model tuning

**Recommendation:** Start with **frozen** (Option 1), then **unfreeze** for final 10-20 epochs.

---

### 7.5 Structure Encoder (EGNN)

**Design:** Train EGNN from scratch on structure graphs with same loss as sequence encoder (binding affinity prediction).

```python
class StructureEncoder(nn.Module):
    """
    E(n)-Equivariant GNN for protein structure graphs
    """
    def __init__(
        self,
        node_dim: int = 33,        # Enhanced node features
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_dim: int = 128
    ):
        super().__init__()
        
        # Node embedding
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        
        # EGNN layers
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(hidden_dim, edge_dim=25)
            for _ in range(num_layers)
        ])
        
        # Global pooling (set2set or attention)
        self.pool = Set2Set(hidden_dim, processing_steps=3)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Set2Set outputs 2*hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.output_dim = output_dim
    
    def forward(self, mhc_graph, peptide_graph):
        """
        Process MHC and peptide graphs separately, then combine
        
        Returns:
            struct_embedding: (batch_size, output_dim)
        """
        # Process MHC graph
        h_mhc = self.node_embed(mhc_graph.x)
        pos_mhc = mhc_graph.pos
        
        for layer in self.egnn_layers:
            h_mhc, pos_mhc = layer(
                h_mhc, pos_mhc, 
                mhc_graph.edge_index, 
                mhc_graph.edge_attr
            )
        
        mhc_emb = self.pool(h_mhc, mhc_graph.batch)  # (batch, 2*hidden_dim)
        
        # Process peptide graph
        h_pep = self.node_embed(peptide_graph.x)
        pos_pep = peptide_graph.pos
        
        for layer in self.egnn_layers:
            h_pep, pos_pep = layer(
                h_pep, pos_pep,
                peptide_graph.edge_index,
                peptide_graph.edge_attr
            )
        
        pep_emb = self.pool(h_pep, peptide_graph.batch)  # (batch, 2*hidden_dim)
        
        # Combine MHC and peptide structure embeddings
        combined = torch.cat([mhc_emb, pep_emb], dim=-1)  # (batch, 4*hidden_dim)
        
        # Project to output dimension
        struct_embedding = self.output_proj(combined)  # (batch, output_dim)
        
        return struct_embedding
```

**Key Design Choices:**

1. **Separate Graph Processing**: MHC and peptide processed independently (different graph sizes)
2. **Set2Set Pooling**: Superior to mean/max pooling for graph-level representations
3. **Concatenation of MHC+Peptide**: Late fusion within structure branch before multimodal fusion

---

### 7.6 Multimodal Fusion Module

**Challenge:** Sequence embedding (1280-dim) and structure embedding (128-dim) have vastly different dimensions and semantic meanings. Naive concatenation may cause **dimension imbalance** (sequence dominates).

**Solution:** **Gated Multimodal Fusion** with learned weighting and cross-modal attention.

#### 7.6.1 Gated Fusion Mechanism

**Objective:** Learn adaptive weighting $\alpha \in [0, 1]$ to balance sequence and structure contributions:

$$
\mathbf{z}_{\text{fused}} = \alpha \cdot \mathbf{z}_{\text{seq}} + (1 - \alpha) \cdot \mathbf{z}_{\text{struct}}
$$

However, this requires **dimension alignment**. First, project both to common dimension:

```python
class GatedMultimodalFusion(nn.Module):
    def __init__(
        self,
        seq_dim: int = 1280,
        struct_dim: int = 128,
        fusion_dim: int = 512
    ):
        super().__init__()
        
        # Project to common dimension
        self.seq_proj = nn.Linear(seq_dim, fusion_dim)
        self.struct_proj = nn.Linear(struct_dim, fusion_dim)
        
        # Gating network: learns adaptive weight
        self.gate_network = nn.Sequential(
            nn.Linear(seq_dim + struct_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, seq_emb, struct_emb):
        """
        Args:
            seq_emb: (batch, seq_dim)
            struct_emb: (batch, struct_dim)
        
        Returns:
            fused: (batch, fusion_dim)
        """
        # Project to common dimension
        seq_proj = self.seq_proj(seq_emb)      # (batch, fusion_dim)
        struct_proj = self.struct_proj(struct_emb)  # (batch, fusion_dim)
        
        # Compute adaptive gate
        gate_input = torch.cat([seq_emb, struct_emb], dim=-1)
        alpha = self.gate_network(gate_input)  # (batch, 1)
        
        # Weighted fusion
        fused = alpha * seq_proj + (1 - alpha) * struct_proj
        
        # Output transformation
        output = self.output_proj(fused)
        
        return output, alpha  # Return gate for analysis
```

**Advantages:**
- **Adaptive Weighting**: Model learns to trust sequence vs. structure per sample
- **Interpretability**: Gate value $\alpha$ reveals which modality drives each prediction
- **Robustness**: Can ignore noisy structure predictions (low pLDDT) by increasing $\alpha$

**Expected Gate Behavior:**
- **High-data alleles** (e.g., HLA-A\*02:01): $\alpha \approx 0.7$ (trust sequence)
- **Rare alleles**: $\alpha \approx 0.4$ (rely more on structure)
- **Low pLDDT structures**: $\alpha \approx 0.9$ (fallback to sequence)

#### 7.6.2 Cross-Modal Attention Fusion

**Enhancement:** Allow sequence and structure features to **attend to each other** before fusion.

```python
class CrossModalAttentionFusion(nn.Module):
    def __init__(
        self,
        seq_dim: int = 1280,
        struct_dim: int = 128,
        fusion_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Project to common dimension
        self.seq_proj = nn.Linear(seq_dim, fusion_dim)
        self.struct_proj = nn.Linear(struct_dim, fusion_dim)
        
        # Cross-modal attention: sequence attends to structure
        self.seq_to_struct_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-modal attention: structure attends to sequence
        self.struct_to_seq_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Gated fusion of attended features
        self.fusion = GatedMultimodalFusion(fusion_dim, fusion_dim, fusion_dim)
    
    def forward(self, seq_emb, struct_emb):
        # Project to common dimension
        seq_proj = self.seq_proj(seq_emb).unsqueeze(1)      # (batch, 1, fusion_dim)
        struct_proj = self.struct_proj(struct_emb).unsqueeze(1)  # (batch, 1, fusion_dim)
        
        # Sequence attends to structure
        seq_attended, _ = self.seq_to_struct_attn(
            query=seq_proj,
            key=struct_proj,
            value=struct_proj
        )  # (batch, 1, fusion_dim)
        
        # Structure attends to sequence
        struct_attended, _ = self.struct_to_seq_attn(
            query=struct_proj,
            key=seq_proj,
            value=seq_proj
        )  # (batch, 1, fusion_dim)
        
        # Gated fusion
        fused, alpha = self.fusion(
            seq_attended.squeeze(1),
            struct_attended.squeeze(1)
        )
        
        return fused, alpha
```

**Advantages:**
- Captures cross-modal dependencies (e.g., "if peptide P2 is Leucine (sequence), check pocket geometry (structure)")
- More expressive than simple concatenation or gating
- Attention weights provide interpretability

**Disadvantages:**
- Higher computational cost (attention operations)
- More parameters to train (potential overfitting)

---

### 7.7 Complete Multimodal Model

**Full Implementation:**

```python
class MultimodalMHCModel(nn.Module):
    """
    Complete multimodal architecture for MHC-peptide binding prediction
    Combines sequence (ESM-2) and structure (EGNN) modalities
    """
    def __init__(
        self,
        seq_checkpoint: str,
        freeze_sequence: bool = True,
        fusion_type: str = 'gated'  # 'gated' or 'cross_attention'
    ):
        super().__init__()
        
        # Sequence encoder (ESM-2 warm-start)
        self.sequence_encoder = SequenceEncoder(
            checkpoint_path=seq_checkpoint,
            freeze=freeze_sequence
        )
        
        # Structure encoder (EGNN)
        self.structure_encoder = StructureEncoder(
            node_dim=33,
            hidden_dim=128,
            num_layers=4,
            output_dim=128
        )
        
        # Multimodal fusion
        if fusion_type == 'gated':
            self.fusion = GatedMultimodalFusion(
                seq_dim=1280,
                struct_dim=128,
                fusion_dim=512
            )
        elif fusion_type == 'cross_attention':
            self.fusion = CrossModalAttentionFusion(
                seq_dim=1280,
                struct_dim=128,
                fusion_dim=512,
                num_heads=8
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(
        self,
        mhc_seq, peptide_seq,      # Sequence inputs (tokenized)
        mhc_graph, peptide_graph    # Structure inputs (PyG graphs)
    ):
        """
        Forward pass through multimodal model
        
        Returns:
            prediction: (batch,) binding affinity predictions
            alpha: (batch,) fusion gate values (interpretability)
        """
        # Extract sequence embedding
        seq_embedding = self.sequence_encoder(mhc_seq, peptide_seq)  # (batch, 1280)
        
        # Extract structure embedding
        struct_embedding = self.structure_encoder(mhc_graph, peptide_graph)  # (batch, 128)
        
        # Multimodal fusion
        fused_embedding, alpha = self.fusion(seq_embedding, struct_embedding)  # (batch, 512)
        
        # Regression
        prediction = self.regression_head(fused_embedding).squeeze(-1)  # (batch,)
        
        return prediction, alpha
```

**Model Statistics:**
- **Total Parameters**: ~661M (ESM-2) + 9M (EGNN) + 2M (fusion + head) = **672M**
- **Trainable** (if seq frozen): ~11M (EGNN + fusion + head)
- **Trainable** (if seq unfrozen): ~672M
- **GPU Memory**: ~8 GB (batch=32, mixed precision)
- **Inference Speed**: ~50 samples/sec (V100 GPU)

---

### 7.8 Handling Missing Structures

**Challenge:** 30-40% of dataset samples lack reliable structures (pLDDT < 50, prediction failures).

**Solution:** **Conditional Fusion** with learned fallback to sequence-only mode.

```python
def forward_with_missing_handling(
    self,
    mhc_seq, peptide_seq,
    mhc_graph=None, peptide_graph=None,
    has_structure=None
):
    """
    Forward pass with graceful handling of missing structures
    
    Args:
        has_structure: (batch,) boolean mask indicating structure availability
    """
    batch_size = mhc_seq.size(0)
    
    # Sequence embedding (always available)
    seq_embedding = self.sequence_encoder(mhc_seq, peptide_seq)
    
    # Initialize structure embedding
    struct_embedding = torch.zeros(batch_size, 128, device=seq_embedding.device)
    
    # Compute structure embedding only for available samples
    if has_structure is not None and has_structure.any():
        struct_indices = has_structure.nonzero(as_tuple=True)[0]
        
        # Subset graphs
        subset_mhc_graph = mhc_graph.index_select(struct_indices)
        subset_pep_graph = peptide_graph.index_select(struct_indices)
        
        # Compute embeddings
        subset_struct_emb = self.structure_encoder(subset_mhc_graph, subset_pep_graph)
        
        # Fill in available embeddings
        struct_embedding[struct_indices] = subset_struct_emb
    
    # Fusion (gate will learn to set alpha≈1 when structure is missing/zero)
    fused_embedding, alpha = self.fusion(seq_embedding, struct_embedding)
    
    # Regression
    prediction = self.regression_head(fused_embedding).squeeze(-1)
    
    return prediction, alpha
```

**Adaptive Gate Behavior:**
- **Structure available** (has_structure=True): Gate learns optimal $\alpha \in [0.3, 0.8]$
- **Structure missing** (has_structure=False, struct_emb=zeros): Gate → 1.0 (pure sequence mode)

**Training Strategy:**
- Train on **both complete and incomplete samples**
- Randomly mask out 20% of structures during training (data augmentation + robustness)
- Model learns to detect and handle unreliable structure inputs

---

### 7.9 Training Configuration

**Objective:** Train multimodal model end-to-end on samples with both sequence and structure.

**Loss Function:**
$$
\mathcal{L} = \text{MSE}(\hat{y}, y) + \lambda_{\text{reg}} \|\theta_{\text{struct}}\|^2 + \lambda_{\text{fusion}} \|\theta_{\text{fusion}}\|^2
$$

**Hyperparameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate (Sequence)** | 1e-6 | Very low (frozen or minimal fine-tuning) |
| **Learning Rate (Structure)** | 5e-4 | Higher (trained from scratch) |
| **Learning Rate (Fusion)** | 1e-4 | Moderate (balances seq/struct) |
| **Weight Decay** | 1e-4 | Regularization |
| **Batch Size** | 32 | GPU memory constraint |
| **Epochs** | 50 | Early stopping patience = 10 |
| **Gradient Clipping** | 1.0 | Stability |
| **LR Schedule** | CosineAnnealingWarmRestarts | Periodic restarts |

**Optimizer Configuration:**

```python
optimizer = torch.optim.AdamW([
    {'params': model.sequence_encoder.parameters(), 'lr': 1e-6},  # Frozen or minimal
    {'params': model.structure_encoder.parameters(), 'lr': 5e-4},  # From scratch
    {'params': model.fusion.parameters(), 'lr': 1e-4},            # Balancing
    {'params': model.regression_head.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)
```

**Training Phases:**

**Phase 1 (Epochs 1-30): Freeze Sequence Encoder**
- Train only structure encoder + fusion + head (~11M params)
- Fast convergence, prevents catastrophic forgetting
- Expected validation RMSE: 0.72-0.75 (pIC50 normalized)

**Phase 2 (Epochs 31-50): Unfreeze Sequence Encoder**
- Fine-tune entire model (~672M params)
- Sequence encoder adapts to structure-aware representations
- Expected validation RMSE: 0.70-0.72 (further improvement)

---

### 7.10 Expected Performance and Ablations

**Performance Targets:**

| Model | Test R² | Test RMSE | Improvement |
|-------|---------|-----------|-------------|
| **ESM-2 650M (sequence)** | 0.7156 | 1.0419 | Baseline |
| **EGNN (structure)** | 0.65-0.70 | 0.90-1.00 | Underperforms |
| **Multimodal (gated)** | **0.74-0.75** | **0.95-1.00** | **+3-5%** |
| **Multimodal (cross-attn)** | **0.75-0.76** | **0.93-0.98** | **+5-6%** |

**Ablation Studies** [[PROPOSED]]:

1. **Fusion Type Comparison:**
   - Concatenation: R² = 0.71-0.72
   - Gated fusion: R² = 0.74-0.75
   - Cross-modal attention: R² = 0.75-0.76

2. **Structure Encoder Variants:**
   - EGNN: R² = 0.74-0.75
   - GVP-GNN: R² = 0.75-0.76 (hypothesis: vectors help)
   - GAT: R² = 0.72-0.73 (less geometric)

3. **Freezing Strategy:**
   - Fully frozen sequence: R² = 0.73-0.74
   - Fine-tuned sequence: R² = 0.75-0.76

4. **Missing Structure Handling:**
   - No missing handling: R² = 0.72 (58% coverage)
   - With fallback: R² = 0.74 (100% coverage)

---

## Summary of Multimodal Fusion Section

This section detailed the **multimodal fusion architecture** combining sequence (ESM-2) and structure (EGNN) modalities:

**Key Contributions:**
1. **Intermediate Fusion Strategy**: Balance pretraining preservation and cross-modal learning
2. **Gated Fusion Mechanism**: Adaptive weighting with learned gate ($\alpha$)
3. **Cross-Modal Attention**: Enhanced expressiveness via attention-based fusion
4. **Missing Structure Handling**: Graceful degradation to sequence-only mode
5. **Phased Training**: Freeze-then-unfreeze for optimal convergence

**Implementation Status:**
- ✅ Architecture design specified
- ✅ PyTorch code provided
- ❌ Training not yet executed
- ❌ Performance targets not yet validated

**Next Steps:**
- Train multimodal model on 45K-58K samples with both modalities
- Validate target R² = 0.74-0.76
- Conduct ablation studies
- Analyze fusion gate interpretability

The next section (Part 6) will detail **training setup and hyperparameters** for all components.

---

**End of Part 5/8**  
*Continue to Part 6/8: Training Setup, Hyperparameters, and Experimental Protocols*
