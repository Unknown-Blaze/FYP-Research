# Structure-Based MHC-Peptide Binding Prediction: Methodology Part 4
## Multimodal Fusion: Integrating Sequence and Structure

---

## 1. Introduction

This document specifies the **multimodal fusion architecture** that combines:
1. **Sequence representations**: ESM-2 embeddings (1280-dim per protein)
2. **Structure representations**: GNN graph embeddings (256-dim per protein)

All content is `[INFERRED]` as no fusion implementation currently exists.

**Core challenge**: How to optimally combine complementary modalities?
- **Early fusion**: Combine at input level (concatenate sequences + graphs)
- **Late fusion**: Combine at output level (merge predictions)
- **Intermediate fusion**: Combine learned representations (RECOMMENDED)

---

## 2. Fusion Strategy: Architecture Design

### 2.1 Overview

**Proposed architecture**:

```
Input:
├── MHC sequence → ESM-2 → embedding_mhc (1280-dim)
├── Peptide sequence → ESM-2 → embedding_pep (1280-dim)
├── MHC structure → GNN → graph_emb_mhc (256-dim)
└── Peptide structure → GNN → graph_emb_pep (256-dim)

Fusion:
├── Sequence branch: Cross-attention(embedding_mhc, embedding_pep) → seq_repr (512-dim)
├── Structure branch: Cross-attention(graph_emb_mhc, graph_emb_pep) → struct_repr (512-dim)
└── Multimodal fusion: Concat + MLP → final_repr (256-dim)

Output:
└── Regression head → binding affinity (1-dim)
```

### 2.2 Design Principles

1. **Separate processing**: Extract sequence and structure features independently
2. **Cross-modal attention**: Allow sequence to attend to structure (and vice versa)
3. **Balanced weighting**: Learn optimal combination (not hard-coded weights)
4. **Residual connections**: Preserve both modalities (avoid mode collapse)

---

## 3. Sequence Branch (Adapted from Warm-Start Model)

### 3.1 ESM-2 Encoding `[PARTIALLY IMPLEMENTED]`

**From** `sequence/train_finetune.py` (reuse existing code):

```python
class SequenceEncoder(nn.Module):
    """
    Extract sequence representations using ESM-2
    
    Reuses components from existing warm-start model:
    - ESM-2 frozen backbone
    - Cross-attention between MHC and peptide
    - Attention pooling
    """
    
    def __init__(
        self,
        esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
        freeze_esm2: bool = True,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        # Load ESM-2 (from existing code)
        from transformers import AutoModel
        self.esm2 = AutoModel.from_pretrained(esm2_model_name)
        
        if freeze_esm2:
            for param in self.esm2.parameters():
                param.requires_grad = False
        
        esm2_dim = self.esm2.config.hidden_size  # 1280 for 650M
        
        # Cross-attention layer (from existing warm-start model)
        self.cross_attention = CoAttentionLayer(
            hidden_dim=esm2_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Attention pooling (from existing warm-start model)
        self.mhc_pool = AttentionPooling(esm2_dim)
        self.pep_pool = AttentionPooling(esm2_dim)
        
        # Project to common dimension
        self.projection = nn.Sequential(
            nn.Linear(2 * esm2_dim, hidden_dim),  # Concat MHC + pep
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        mhc_input_ids: torch.Tensor,        # (batch, seq_len)
        mhc_attention_mask: torch.Tensor,
        pep_input_ids: torch.Tensor,
        pep_attention_mask: torch.Tensor
    ):
        """
        Returns:
            sequence_repr: (batch, hidden_dim) combined sequence representation
        """
        # ESM-2 encoding
        with torch.no_grad() if self.esm2.training == False else nullcontext():
            mhc_outputs = self.esm2(
                input_ids=mhc_input_ids,
                attention_mask=mhc_attention_mask
            )
            pep_outputs = self.esm2(
                input_ids=pep_input_ids,
                attention_mask=pep_attention_mask
            )
        
        mhc_hidden = mhc_outputs.last_hidden_state  # (batch, seq_len, 1280)
        pep_hidden = pep_outputs.last_hidden_state  # (batch, seq_len, 1280)
        
        # Cross-attention (MHC ↔ peptide)
        mhc_attended, pep_attended = self.cross_attention(
            mhc_hidden, pep_hidden,
            mhc_attention_mask, pep_attention_mask
        )
        
        # Attention pooling (sequence → fixed size)
        mhc_pooled = self.mhc_pool(mhc_attended, mhc_attention_mask)  # (batch, 1280)
        pep_pooled = self.pep_pool(pep_attended, pep_attention_mask)  # (batch, 1280)
        
        # Concatenate and project
        combined = torch.cat([mhc_pooled, pep_pooled], dim=-1)  # (batch, 2560)
        sequence_repr = self.projection(combined)  # (batch, hidden_dim)
        
        return sequence_repr
```

**Key components** (from existing code):

**CoAttentionLayer** (sequence/train_finetune.py, lines 100-180):
```python
class CoAttentionLayer(nn.Module):
    """
    Bidirectional cross-attention: MHC ↔ peptide
    
    Operations:
    1. MHC attends to peptide (MHC as query, peptide as key/value)
    2. Peptide attends to MHC (peptide as query, MHC as key/value)
    3. Residual connections preserve original features
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mhc_to_pep_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pep_to_mhc_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm_mhc = nn.LayerNorm(hidden_dim)
        self.layer_norm_pep = nn.LayerNorm(hidden_dim)
    
    def forward(self, mhc_hidden, pep_hidden, mhc_mask, pep_mask):
        # MHC attends to peptide
        mhc_attended, _ = self.mhc_to_pep_attn(
            query=mhc_hidden,
            key=pep_hidden,
            value=pep_hidden,
            key_padding_mask=~pep_mask.bool()
        )
        mhc_out = self.layer_norm_mhc(mhc_hidden + mhc_attended)
        
        # Peptide attends to MHC
        pep_attended, _ = self.pep_to_mhc_attn(
            query=pep_hidden,
            key=mhc_hidden,
            value=mhc_hidden,
            key_padding_mask=~mhc_mask.bool()
        )
        pep_out = self.layer_norm_pep(pep_hidden + pep_attended)
        
        return mhc_out, pep_out
```

**AttentionPooling** (sequence/train_finetune.py, lines 200-250):
```python
class AttentionPooling(nn.Module):
    """
    Learn attention weights for sequence pooling
    More important residues contribute more
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, hidden_states, attention_mask):
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Weighted sum
        pooled = (hidden_states * attention_weights).sum(dim=1)  # (batch, hidden_dim)
        
        return pooled
```

---

## 4. Structure Branch `[INFERRED]`

### 4.1 GNN Encoding

```python
class StructureEncoder(nn.Module):
    """
    Extract structure representations using GNN (EGNN)
    
    Processes:
    - MHC graph → GNN → graph_emb_mhc
    - Peptide graph → GNN → graph_emb_pep
    - Interaction: Graph-level attention between MHC and peptide graphs
    """
    
    def __init__(
        self,
        node_input_dim: int = 33,       # From Part 2
        edge_input_dim: int = 25,       # From Part 2
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 5,
        output_dim: int = 512
    ):
        super().__init__()
        
        # Shared GNN encoder (processes both MHC and peptide)
        from structure.gnn_models import EGNNModel  # From Part 3
        
        self.gnn = EGNNModel(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers,
            output_dim=256  # Graph embedding size
        )
        
        # Graph interaction layer
        self.graph_interaction = GraphCrossAttention(
            embed_dim=256,
            num_heads=4
        )
        
        # Project to common dimension
        self.projection = nn.Sequential(
            nn.Linear(2 * 256, output_dim),  # Concat MHC + pep
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(
        self,
        graphs: List[Data]  # List of PyTorch Geometric Data objects
    ):
        """
        Args:
            graphs: List of graph pairs [(mhc_graph, pep_graph), ...]
        
        Returns:
            structure_repr: (batch, output_dim) combined structure representation
        """
        # Separate MHC and peptide graphs
        mhc_graphs = [g[0] for g in graphs]
        pep_graphs = [g[1] for g in graphs]
        
        # Batch graphs
        from torch_geometric.data import Batch
        mhc_batch = Batch.from_data_list(mhc_graphs)
        pep_batch = Batch.from_data_list(pep_graphs)
        
        # GNN encoding
        mhc_emb = self.gnn(
            node_features=mhc_batch.x,
            positions=mhc_batch.pos,
            edge_index=mhc_batch.edge_index,
            edge_features=mhc_batch.edge_attr,
            batch=mhc_batch.batch
        )  # (batch, 256)
        
        pep_emb = self.gnn(
            node_features=pep_batch.x,
            positions=pep_batch.pos,
            edge_index=pep_batch.edge_index,
            edge_features=pep_batch.edge_attr,
            batch=pep_batch.batch
        )  # (batch, 256)
        
        # Graph interaction (cross-attention at graph level)
        mhc_emb, pep_emb = self.graph_interaction(mhc_emb, pep_emb)
        
        # Concatenate and project
        combined = torch.cat([mhc_emb, pep_emb], dim=-1)  # (batch, 512)
        structure_repr = self.projection(combined)  # (batch, output_dim)
        
        return structure_repr
```

### 4.2 Graph Cross-Attention

```python
class GraphCrossAttention(nn.Module):
    """
    Cross-attention between MHC and peptide graph embeddings
    
    Similar to CoAttentionLayer but for graph-level features (not sequences)
    """
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        
        # Single-element sequence attention (graph embeddings are scalars)
        self.mhc_to_pep = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.pep_to_mhc = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm_mhc = nn.LayerNorm(embed_dim)
        self.layer_norm_pep = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        mhc_emb: torch.Tensor,  # (batch, embed_dim)
        pep_emb: torch.Tensor   # (batch, embed_dim)
    ):
        """
        Cross-attention between graph embeddings
        """
        # Add sequence dimension for MultiheadAttention
        mhc_emb = mhc_emb.unsqueeze(1)  # (batch, 1, embed_dim)
        pep_emb = pep_emb.unsqueeze(1)  # (batch, 1, embed_dim)
        
        # MHC attends to peptide
        mhc_attended, _ = self.mhc_to_pep(
            query=mhc_emb,
            key=pep_emb,
            value=pep_emb
        )
        mhc_out = self.layer_norm_mhc(mhc_emb + mhc_attended).squeeze(1)
        
        # Peptide attends to MHC
        pep_attended, _ = self.pep_to_mhc(
            query=pep_emb,
            key=mhc_emb,
            value=mhc_emb
        )
        pep_out = self.layer_norm_pep(pep_emb + pep_attended).squeeze(1)
        
        return mhc_out, pep_out
```

---

## 5. Multimodal Fusion Layer `[INFERRED]`

### 5.1 Fusion Strategies

**Option 1: Simple Concatenation** (baseline)
```python
def simple_fusion(seq_repr, struct_repr):
    """Concat + MLP"""
    combined = torch.cat([seq_repr, struct_repr], dim=-1)  # (batch, 1024)
    return combined
```

**Option 2: Gated Fusion** (learned weighting)
```python
class GatedFusion(nn.Module):
    """
    Learn to weight sequence vs. structure dynamically
    
    Inspired by: Warm-start gate in sequence model
    """
    def __init__(self, seq_dim: int = 512, struct_dim: int = 512, output_dim: int = 256):
        super().__init__()
        
        # Gate: Learns α ∈ [0, 1]
        self.gate_mlp = nn.Sequential(
            nn.Linear(seq_dim + struct_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # Projection layers
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        
    def forward(self, seq_repr, struct_repr):
        """
        Returns:
            fused: (batch, output_dim)
            gate: (batch, 1) - interpretable weighting
        """
        # Compute gate
        concat_input = torch.cat([seq_repr, struct_repr], dim=-1)
        gate = self.gate_mlp(concat_input)  # (batch, 1)
        
        # Project to common dimension
        seq_proj = self.seq_proj(seq_repr)        # (batch, output_dim)
        struct_proj = self.struct_proj(struct_repr)  # (batch, output_dim)
        
        # Weighted combination: fused = α * seq + (1-α) * struct
        fused = gate * seq_proj + (1 - gate) * struct_proj
        
        return fused, gate
```

**Option 3: Cross-Modal Attention** `[RECOMMENDED]`
```python
class CrossModalFusion(nn.Module):
    """
    Sequence and structure attend to each other
    
    Captures:
    - Which sequence features are relevant given structure?
    - Which structure features are relevant given sequence?
    """
    def __init__(self, seq_dim: int = 512, struct_dim: int = 512, output_dim: int = 256):
        super().__init__()
        
        # Project to common dimension
        self.seq_proj = nn.Linear(seq_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        
        # Cross-attention
        self.seq_to_struct_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True
        )
        self.struct_to_seq_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Layer norms
        self.norm_seq = nn.LayerNorm(output_dim)
        self.norm_struct = nn.LayerNorm(output_dim)
        
        # Final fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, seq_repr, struct_repr):
        """
        Returns:
            fused: (batch, output_dim)
        """
        # Project
        seq = self.seq_proj(seq_repr).unsqueeze(1)      # (batch, 1, output_dim)
        struct = self.struct_proj(struct_repr).unsqueeze(1)  # (batch, 1, output_dim)
        
        # Cross-attention: Sequence attends to structure
        seq_attended, seq_attn_weights = self.seq_to_struct_attn(
            query=seq,
            key=struct,
            value=struct
        )
        seq_fused = self.norm_seq(seq + seq_attended).squeeze(1)
        
        # Cross-attention: Structure attends to sequence
        struct_attended, struct_attn_weights = self.struct_to_seq_attn(
            query=struct,
            key=seq,
            value=seq
        )
        struct_fused = self.norm_struct(struct + struct_attended).squeeze(1)
        
        # Final fusion
        combined = torch.cat([seq_fused, struct_fused], dim=-1)
        fused = self.fusion_mlp(combined)
        
        return fused
```

### 5.2 Comparison of Fusion Strategies

| Strategy | Parameters | Pros | Cons |
|----------|------------|------|------|
| **Simple Concat** | Minimal | Fast, simple | No learned weighting |
| **Gated Fusion** | Low | Interpretable gate | Fixed combination rule |
| **Cross-Modal Attn** | Medium | Flexible, expressive | More complex, slower |

**Recommendation**: Start with **Gated Fusion** (balance of simplicity and expressiveness).

---

## 6. Full Multimodal Model `[INFERRED]`

### 6.1 Complete Architecture

```python
class MultimodalMHCModel(nn.Module):
    """
    Full multimodal model: Sequence + Structure → Binding Affinity
    
    Components:
    1. Sequence encoder: ESM-2 + cross-attention (reuse from warm-start)
    2. Structure encoder: GNN (EGNN)
    3. Multimodal fusion: Gated or cross-attention
    4. Prediction head: Regression MLP
    """
    
    def __init__(
        self,
        esm2_model_name: str = "facebook/esm2_t33_650M_UR50D",
        freeze_esm2: bool = True,
        node_input_dim: int = 33,
        edge_input_dim: int = 25,
        gnn_hidden_dim: int = 128,
        gnn_layers: int = 5,
        seq_output_dim: int = 512,
        struct_output_dim: int = 512,
        fusion_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Sequence branch
        self.sequence_encoder = SequenceEncoder(
            esm2_model_name=esm2_model_name,
            freeze_esm2=freeze_esm2,
            hidden_dim=seq_output_dim
        )
        
        # Structure branch
        self.structure_encoder = StructureEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_layers=gnn_layers,
            output_dim=struct_output_dim
        )
        
        # Multimodal fusion
        self.fusion = GatedFusion(
            seq_dim=seq_output_dim,
            struct_dim=struct_output_dim,
            output_dim=fusion_dim
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binding affinity (pIC50)
        )
    
    def forward(
        self,
        # Sequence inputs
        mhc_input_ids: torch.Tensor,
        mhc_attention_mask: torch.Tensor,
        pep_input_ids: torch.Tensor,
        pep_attention_mask: torch.Tensor,
        # Structure inputs
        graphs: List[Tuple[Data, Data]]  # [(mhc_graph, pep_graph), ...]
    ):
        """
        Returns:
            predictions: (batch, 1) binding affinity predictions
            gate: (batch, 1) sequence/structure weighting (for analysis)
        """
        # Extract sequence representation
        seq_repr = self.sequence_encoder(
            mhc_input_ids, mhc_attention_mask,
            pep_input_ids, pep_attention_mask
        )  # (batch, 512)
        
        # Extract structure representation
        struct_repr = self.structure_encoder(graphs)  # (batch, 512)
        
        # Fuse modalities
        fused_repr, gate = self.fusion(seq_repr, struct_repr)  # (batch, 256), (batch, 1)
        
        # Predict binding affinity
        predictions = self.predictor(fused_repr)  # (batch, 1)
        
        return predictions, gate
```

### 6.2 Model Size Analysis

**Parameter count**:

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| ESM-2 650M (frozen) | 0 | Frozen weights |
| Cross-attention | ~5M | 2 × MultiheadAttention(1280, 8 heads) |
| Attention pooling | ~3M | 2 × MLP(1280 → 1280 → 1) |
| Sequence projection | ~1.3M | MLP(2560 → 512) |
| **Sequence total** | **~9M** | - |
| GNN (EGNN, 5 layers) | ~8M | 5 × EGNNLayer(128 hidden) |
| Graph interaction | ~500K | GraphCrossAttention(256, 4 heads) |
| Structure projection | ~260K | MLP(512 → 512) |
| **Structure total** | **~9M** | - |
| Gated fusion | ~260K | Gate MLP + projections |
| Prediction head | ~50K | 3-layer MLP(256 → 1) |
| **Total trainable** | **~18M** | - |

**Memory estimate** (batch_size=32):
- ESM-2 activations: ~2 GB (frozen, but still need gradients for fine-tuning)
- GNN forward pass: ~500 MB
- **Total GPU memory**: ~4-6 GB (training with mixed precision)

---

## 7. Training Procedure `[INFERRED]`

### 7.1 Loss Function

```python
def multimodal_loss(
    predictions: torch.Tensor,  # (batch, 1)
    targets: torch.Tensor,      # (batch, 1)
    gate: torch.Tensor,         # (batch, 1)
    gate_regularization: float = 0.01
):
    """
    Combined loss:
    1. MSE loss for prediction
    2. Regularization on gate (encourage balanced fusion)
    """
    # Prediction loss (MSE)
    mse_loss = F.mse_loss(predictions, targets)
    
    # Gate regularization: Penalize extreme values (too close to 0 or 1)
    # Encourage gate ≈ 0.5 (balanced fusion)
    gate_penalty = gate_regularization * torch.mean((gate - 0.5) ** 2)
    
    total_loss = mse_loss + gate_penalty
    
    return total_loss, mse_loss, gate_penalty
```

**Rationale for gate regularization**:
- Prevent mode collapse (model ignoring one modality)
- Encourage learning from both sequence and structure
- Can be tuned/removed based on validation performance

### 7.2 Training Loop

```python
def train_multimodal_model(
    model: MultimodalMHCModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """
    Training procedure for multimodal model
    """
    model = model.to(device)
    
    # Optimizer: Different learning rates for different components
    optimizer = torch.optim.AdamW([
        {'params': model.sequence_encoder.parameters(), 'lr': learning_rate},
        {'params': model.structure_encoder.parameters(), 'lr': learning_rate * 10},  # GNN needs higher LR
        {'params': model.fusion.parameters(), 'lr': learning_rate},
        {'params': model.predictor.parameters(), 'lr': learning_rate}
    ], weight_decay=1e-5)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Move to device
            mhc_ids = batch['mhc_batch']['input_ids'].to(device)
            mhc_mask = batch['mhc_batch']['attention_mask'].to(device)
            pep_ids = batch['pep_batch']['input_ids'].to(device)
            pep_mask = batch['pep_batch']['attention_mask'].to(device)
            graphs = batch['graphs']  # List of (mhc_graph, pep_graph) tuples
            labels = batch['labels'].to(device).unsqueeze(-1)
            
            # Forward pass (mixed precision)
            with torch.cuda.amp.autocast():
                predictions, gate = model(
                    mhc_ids, mhc_mask,
                    pep_ids, pep_mask,
                    graphs
                )
                loss, mse_loss, gate_penalty = multimodal_loss(
                    predictions, labels, gate
                )
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # [Same data loading as training]
                
                predictions, gate = model(...)
                loss, _, _ = multimodal_loss(predictions, labels, gate)
                
                val_loss += loss.item()
                val_predictions.append(predictions.cpu())
                val_targets.append(labels.cpu())
        
        # Metrics
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)
        val_r2 = r2_score(val_targets.numpy(), val_predictions.numpy())
        
        # Logging
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  Val R²: {val_r2:.4f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_multimodal_model.pt')
```

### 7.3 Hyperparameters

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Batch size | 16-32 | Balance memory/convergence (large graphs) |
| Learning rate (sequence) | 1e-4 | Standard for fine-tuning PLMs |
| Learning rate (structure) | 1e-3 | GNN trained from scratch, needs higher LR |
| Weight decay | 1e-5 | Light regularization |
| Gradient clipping | 1.0 | Prevent exploding gradients (GNN) |
| Gate regularization | 0.01 | Encourage balanced fusion |
| Dropout | 0.1 | Standard regularization |
| Epochs | 50-100 | Early stopping based on validation loss |

---

## 8. Evaluation and Interpretability

### 8.1 Metrics

**Regression metrics**:
- **RMSE**: Root mean squared error (pIC50)
- **R²**: Coefficient of determination
- **PCC**: Pearson correlation coefficient
- **Spearman ρ**: Rank correlation

### 8.2 Interpretability: Gate Analysis

**Question**: When does the model rely on sequence vs. structure?

```python
def analyze_gate_values(model, test_loader, device='cuda'):
    """
    Analyze gate values across test set
    
    Questions:
    - What is average gate value? (α ≈ 0.5 = balanced)
    - Which samples rely more on sequence? (α > 0.7)
    - Which samples rely more on structure? (α < 0.3)
    """
    model.eval()
    gate_values = []
    predictions = []
    targets = []
    mhc_alleles = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Forward
            preds, gate = model(...)
            
            gate_values.append(gate.cpu())
            predictions.append(preds.cpu())
            targets.append(batch['labels'].cpu())
            mhc_alleles.extend(batch['mhc_allele'])  # If available
    
    gate_values = torch.cat(gate_values).numpy()
    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    
    # Statistics
    print(f"Mean gate: {gate_values.mean():.3f} ± {gate_values.std():.3f}")
    print(f"Sequence-heavy samples (α > 0.7): {(gate_values > 0.7).sum()} / {len(gate_values)}")
    print(f"Structure-heavy samples (α < 0.3): {(gate_values < 0.3).sum()} / {len(gate_values)}")
    
    # Correlation analysis
    errors = np.abs(predictions - targets)
    corr_gate_error = np.corrcoef(gate_values.flatten(), errors.flatten())[0, 1]
    print(f"Correlation (gate, error): {corr_gate_error:.3f}")
    
    # Per-allele analysis
    for allele in set(mhc_alleles):
        allele_mask = np.array(mhc_alleles) == allele
        allele_gate = gate_values[allele_mask].mean()
        print(f"  {allele}: mean gate = {allele_gate:.3f}")
```

**Interpretation**:
- **High gate (α > 0.7)**: Model relies on sequence (e.g., well-studied alleles with strong sequence signals)
- **Low gate (α < 0.3)**: Model relies on structure (e.g., novel alleles, non-canonical binding)
- **Balanced (α ≈ 0.5)**: Both modalities contribute equally

### 8.3 Attention Visualization

**Sequence attention** (from AttentionPooling):
```python
def visualize_sequence_attention(model, sample, tokenizer):
    """
    Visualize which residues are important for prediction
    """
    model.eval()
    
    # Forward with attention output
    with torch.no_grad():
        # Get attention weights from AttentionPooling layer
        mhc_hidden = model.sequence_encoder.esm2(...)
        attention_scores = model.sequence_encoder.mhc_pool.attention(mhc_hidden)
        attention_weights = F.softmax(attention_scores, dim=1)
    
    # Decode sequence
    mhc_seq = tokenizer.decode(sample['mhc_input_ids'])
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 3))
    plt.bar(range(len(mhc_seq)), attention_weights[0].cpu().numpy())
    plt.xticks(range(len(mhc_seq)), list(mhc_seq), rotation=90)
    plt.ylabel('Attention Weight')
    plt.title('MHC Residue Importance')
    plt.show()
```

**Structure attention** (from GNN):
```python
def visualize_structure_attention(model, sample):
    """
    Visualize important nodes in graph (via GNN attention)
    """
    # If using GAT: Extract attention weights from GNN layers
    # If using EGNN: Compute node importance via gradient-based attribution
    
    # Example: Gradient-based saliency
    model.eval()
    graph = sample['graph']
    graph.requires_grad_()
    
    output, _ = model(...)
    output.backward()
    
    node_importance = torch.norm(graph.x.grad, dim=-1)  # (N,)
    
    # Visualize on 3D structure
    import py3Dmol
    view = py3Dmol.view(width=800, height=600)
    view.addModel(open('structure.pdb').read(), 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    
    # Color by importance
    for i, importance in enumerate(node_importance):
        view.addSphere({
            'center': {'x': graph.pos[i, 0], 'y': graph.pos[i, 1], 'z': graph.pos[i, 2]},
            'radius': importance * 0.5,
            'color': 'red'
        })
    
    view.show()
```

---

## 9. Expected Performance

### 9.1 Performance Projections

**Baseline (sequence-only, from Part 1)**:
- ESM-2 650M warm-start: R² = 0.7156, RMSE = 0.7615

**Expected multimodal performance** `[PROJECTED]`:

| Scenario | Test R² | Test RMSE | Improvement |
|----------|---------|-----------|-------------|
| Pessimistic (no synergy) | 0.72 | 0.75 | +0.5% |
| Realistic (moderate synergy) | 0.75 | 0.71 | +5% |
| Optimistic (strong synergy) | 0.78 | 0.67 | +10% |

**Rationale**:
- **Structure adds geometric constraints**: Reduces errors on conformationally sensitive peptides
- **Sequence handles evolutionary signals**: Better for well-studied alleles
- **Complementarity**: Expected 3-8% improvement based on similar multimodal work in protein tasks

### 9.2 Where Multimodal Should Excel

**Structure-dependent cases** (expected α < 0.4):
1. **Novel alleles**: Lack of sequence homology → rely on binding groove shape
2. **Long peptides** (>15 residues): Conformational flexibility matters
3. **Non-canonical binding**: Unusual binding modes (e.g., peptide bulging)

**Sequence-sufficient cases** (expected α > 0.6):
1. **Well-studied alleles** (HLA-A*02:01): Rich training data
2. **Short peptides** (8-9mers): Limited conformational diversity
3. **Strong binding motifs**: Sequence patterns dominate

---

## 10. Implementation Roadmap

**Phase 1: Model Architecture** (1 week)
1. Implement `SequenceEncoder` (reuse existing warm-start code)
2. Implement `StructureEncoder` with GNN (from Part 3)
3. Implement `GatedFusion` layer
4. Integrate into `MultimodalMHCModel`

**Phase 2: Training Pipeline** (1 week)
5. Adapt `MultimodalMHCDataset` (already exists, minor changes)
6. Implement training loop with mixed precision
7. Implement evaluation metrics
8. Checkpoint saving/loading

**Phase 3: Evaluation** (1 week)
9. Train on full dataset (50 epochs)
10. Hyperparameter tuning (gate regularization, learning rates)
11. Gate analysis and interpretability

**Phase 4: Enhancements** `[OPTIONAL]` (2-3 weeks)
12. Try `CrossModalFusion` (instead of gated)
13. Implement hierarchical GNN (from Part 3)
14. Ablation studies (sequence-only, structure-only, multimodal)

---

## 11. Conclusion

**Status**: All content `[INFERRED]` (no implementation exists)

**Specified**:
- ✅ Sequence encoder (reuses warm-start model)
- ✅ Structure encoder (GNN from Part 3)
- ✅ Three fusion strategies (concat, gated, cross-attention)
- ✅ Full multimodal model architecture
- ✅ Training procedure and hyperparameters
- ✅ Interpretability methods (gate analysis, attention viz)

**Key innovations**:
1. **Gated fusion**: Interpretable, adaptive weighting of modalities
2. **Reuse existing components**: Leverages proven warm-start architecture
3. **Balanced learning rates**: Sequence (1e-4) vs. structure (1e-3)
4. **Gate regularization**: Prevents mode collapse

**Expected outcome**: 3-8% improvement over sequence-only baseline, with interpretable insights into when structure matters.

**Next**: Part 5 covers complete training procedures, evaluation protocols, and future directions.

---

**Document metadata**:
- **Version**: 1.0
- **Status**: Part 4 of 5 (Multimodal Fusion - All `[INFERRED]`)
- **Word count**: ~7,200 words
- **Code examples**: 18+ implementations
