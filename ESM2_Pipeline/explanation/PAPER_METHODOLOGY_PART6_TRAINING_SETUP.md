# COMPREHENSIVE MULTIMODAL MHC-PEPTIDE BINDING PREDICTION METHODOLOGY

## PART 6/8: TRAINING SETUP, HYPERPARAMETERS, AND EXPERIMENTAL PROTOCOLS

---

## 8. Comprehensive Training Configuration

### 8.1 Hardware and Software Environment

**Computing Infrastructure:**

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA A100 40GB (primary) / V100 32GB (secondary) |
| **CPU** | Intel Xeon Gold 6248R (48 cores) |
| **RAM** | 256 GB DDR4 |
| **Storage** | 2 TB NVMe SSD (model checkpoints, graphs) |
| **OS** | Ubuntu 22.04 LTS |
| **CUDA** | 11.8 |
| **cuDNN** | 8.6 |

**Software Stack:**

| Framework/Library | Version | Purpose |
|-------------------|---------|---------|
| **Python** | 3.10.12 | Programming language |
| **PyTorch** | 2.0.1 | Deep learning framework |
| **PyTorch Geometric** | 2.3.1 | Graph neural networks |
| **Transformers** | 4.30.2 | ESM-2 protein language models |
| **ESM** | 2.0.0 | Meta AI ESM models |
| **NumPy** | 1.24.3 | Numerical computing |
| **Pandas** | 2.0.3 | Data manipulation |
| **scikit-learn** | 1.3.0 | Data splitting, metrics |
| **Matplotlib** | 3.7.2 | Visualization |
| **TensorBoard** | 2.13.0 | Training logging |
| **ColabFold** | 1.5.2 | Structure prediction |
| **Weights & Biases** | [[OPTIONAL]] | Experiment tracking |

**Installation:**

```bash
# Create conda environment
conda create -n mhc_multimodal python=3.10 -y
conda activate mhc_multimodal

# PyTorch with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric==2.3.1
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Transformers and ESM
pip install transformers==4.30.2
pip install fair-esm==2.0.0

# Scientific stack
pip install numpy pandas scikit-learn matplotlib tensorboard

# ColabFold (separate environment for structure prediction)
conda create -n colabfold python=3.9 -y
conda activate colabfold
conda install -c conda-forge -c bioconda colabfold
```

---

### 8.2 Dataset Preparation and Augmentation

**Data Loading Pipeline:**

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalMHCDataset(Dataset):
    """
    Dataset combining sequence and structure modalities
    Handles missing structures gracefully
    """
    def __init__(
        self,
        csv_file: str,
        structure_dir: str,
        tokenizer,
        max_length: int = 512,
        transform=None
    ):
        self.data = pd.read_csv(csv_file)
        self.structure_dir = structure_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        
        # Load structure availability mapping
        self.structure_mapping = self._load_structure_mapping()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Sequence data (always available)
        mhc_seq = row['mhc_sequence']
        pep_seq = row['peptide']
        label = row['binding_affinity_nM']  # Will be transformed to pIC50
        
        # Tokenize sequences
        mhc_tokens = self.tokenizer(
            mhc_seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        pep_tokens = self.tokenizer(
            pep_seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Structure data (may be missing)
        has_structure = idx in self.structure_mapping
        
        if has_structure:
            mhc_graph_file = self.structure_mapping[idx]['mhc_graph']
            pep_graph_file = self.structure_mapping[idx]['pep_graph']
            
            mhc_graph = torch.load(mhc_graph_file)
            pep_graph = torch.load(pep_graph_file)
        else:
            # Create dummy graphs (will be masked in forward pass)
            mhc_graph = self._create_dummy_graph()
            pep_graph = self._create_dummy_graph()
        
        return {
            'mhc_tokens': mhc_tokens,
            'pep_tokens': pep_tokens,
            'mhc_graph': mhc_graph,
            'pep_graph': pep_graph,
            'label': torch.tensor(label, dtype=torch.float32),
            'has_structure': torch.tensor(has_structure, dtype=torch.bool)
        }
```

**Data Augmentation Strategies:**

1. **Sequence Augmentation** (NOT APPLIED):
   - Random amino acid substitution: Too risky (single mutation can abolish binding)
   - Back-translation: Not applicable (protein sequences, not natural language)
   - **Verdict**: No augmentation for sequences (preserve biological accuracy)

2. **Structure Augmentation** (APPLIED):
   ```python
   def augment_structure(graph, training=True):
       if not training:
           return graph
       
       # Random rotation (test E(3)-equivariance)
       if random.random() < 0.5:
           R = random_rotation_matrix()
           graph.pos = graph.pos @ R.T
       
       # Random translation
       if random.random() < 0.5:
           t = torch.randn(3) * 1.0  # Shift by ~1 Å
           graph.pos = graph.pos + t
       
       # Gaussian noise (simulate prediction uncertainty)
       if random.random() < 0.3:
           noise = torch.randn_like(graph.pos) * 0.1  # σ = 0.1 Å
           graph.pos = graph.pos + noise
       
       return graph
   ```

3. **Label Smoothing** (APPLIED):
   ```python
   # Smooth pIC50 labels with Gaussian noise (σ = 0.05)
   if training:
       label = label + torch.randn_like(label) * 0.05
       label = label.clamp(min=0, max=12)  # Valid pIC50 range
   ```

---

### 8.3 Hyperparameter Configuration

#### 8.3.1 Sequence Model (ESM-2 Warm-Start)

**Configuration File:** `sequence/config.yaml`

```yaml
# Model architecture
model_name: "facebook/esm2_t33_650M_UR50D"
tokenizer_name: "facebook/esm2_t33_650M_UR50D"
freeze_esm: false
num_ca_layers: 1
num_attention_heads: 8
ca_dropout: 0.15
warm_start_gate: 0.1

# Training
epochs: 100
batch_size: 32
accum_steps: 1
lr: 0.00001
lr_attention_multiplier: 10.0
scheduler_patience: 3
clip_norm: 1.0
early_stopping_patience: 10
use_amp: true
num_workers: 12

# Data preprocessing
convert_to_pic50: true
normalize: true
seed: 42
test_size: 0.1
valid_size: 0.1
```

**Key Hyperparameters Explained:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr: 0.00001` | 1e-5 | Conservative for fine-tuning 650M params |
| `lr_attention_multiplier: 10.0` | 10× base LR | Randomly initialized cross-attention needs faster learning |
| `warm_start_gate: 0.1` | 10% attention, 90% ESM | Preserves pretrained knowledge initially |
| `ca_dropout: 0.15` | 15% | Regularization for attention layers |
| `scheduler_patience: 3` | 3 epochs | Reduce LR if plateau for 3 epochs |
| `early_stopping_patience: 10` | 10 epochs | Stop if no improvement for 10 epochs |
| `use_amp: true` | Mixed precision | 2× speedup, 50% memory reduction |

**Learning Rate Schedule:**

```python
optimizer = torch.optim.AdamW([
    {'params': esm_model.parameters(), 'lr': 1e-5},
    {'params': attention_layers.parameters(), 'lr': 1e-4}  # 10× higher
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)
```

#### 8.3.2 Structure Model (EGNN)

**Configuration File:** `structure/config_egnn.yaml` [[PROPOSED]]

```yaml
# Model architecture
node_dim: 33  # Amino acid + physicochemical + pLDDT
edge_dim: 25  # Distance + RBF + angles
hidden_dim: 128
num_layers: 4
output_dim: 128
pool_type: "set2set"  # Options: mean, attention, set2set

# Training
epochs: 100
batch_size: 64  # Larger than sequence (graphs are smaller)
lr: 0.0005
weight_decay: 0.0001
scheduler: "cosine"  # CosineAnnealingWarmRestarts
clip_norm: 1.0
early_stopping_patience: 15

# Data
contact_distance: 10.0  # Å cutoff for edges
augmentation: true
augment_rotation: true
augment_translation: true
augment_noise_std: 0.1
```

**Key Differences from Sequence Model:**

- **Higher Learning Rate** (5e-4 vs. 1e-5): No pretraining, train from scratch
- **Larger Batch Size** (64 vs. 32): Graphs fit in memory better than transformer sequences
- **Cosine Schedule**: Better for GNNs (smooth annealing)
- **Longer Patience** (15 vs. 10): GNNs converge slower than fine-tuned PLMs

#### 8.3.3 Multimodal Fusion Model

**Configuration File:** `multimodal/config_fusion.yaml` [[PROPOSED]]

```yaml
# Sequence encoder (reuse pretrained)
sequence_checkpoint: "./checkpoints/checkpoints_esm2_warmstart/best_model.pt"
freeze_sequence: true  # Phase 1: freeze, Phase 2: unfreeze

# Structure encoder (train from scratch)
structure_config:
  node_dim: 33
  hidden_dim: 128
  num_layers: 4
  output_dim: 128

# Fusion module
fusion_type: "cross_attention"  # Options: gated, cross_attention, bilinear
seq_dim: 1280
struct_dim: 128
fusion_dim: 512
num_fusion_heads: 8
fusion_dropout: 0.15

# Training phases
phase1_epochs: 30  # Freeze sequence encoder
phase2_epochs: 20  # Unfreeze sequence encoder

# Optimization
lr_sequence: 0.000001  # 1e-6 (minimal fine-tuning)
lr_structure: 0.0005   # 5e-4 (from scratch)
lr_fusion: 0.0001      # 1e-4 (balanced)
weight_decay: 0.0001
batch_size: 32
gradient_accumulation: 1
clip_norm: 1.0

# Scheduler
scheduler: "cosine_warmup"
warmup_steps: 500
max_lr: [1e-6, 5e-4, 1e-4]  # Per parameter group
min_lr: [1e-8, 1e-6, 1e-6]

# Early stopping
early_stopping_patience: 10
min_delta: 0.001  # Minimum improvement threshold
```

**Phased Training Strategy:**

**Phase 1: Frozen Sequence Encoder (Epochs 1-30)**
```python
# Freeze ESM-2
for param in model.sequence_encoder.parameters():
    param.requires_grad = False

# Only train structure + fusion
trainable_params = (
    list(model.structure_encoder.parameters()) +
    list(model.fusion.parameters()) +
    list(model.regression_head.parameters())
)

optimizer = torch.optim.AdamW([
    {'params': model.structure_encoder.parameters(), 'lr': 5e-4},
    {'params': model.fusion.parameters(), 'lr': 1e-4},
    {'params': model.regression_head.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)
```

**Phase 2: Unfrozen Sequence Encoder (Epochs 31-50)**
```python
# Unfreeze ESM-2
for param in model.sequence_encoder.parameters():
    param.requires_grad = True

# Fine-tune all components
optimizer = torch.optim.AdamW([
    {'params': model.sequence_encoder.parameters(), 'lr': 1e-6},  # Very low
    {'params': model.structure_encoder.parameters(), 'lr': 5e-4},
    {'params': model.fusion.parameters(), 'lr': 1e-4},
    {'params': model.regression_head.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)
```

---

### 8.4 Loss Functions and Objectives

**Primary Loss:** Mean Squared Error (MSE) on normalized pIC50

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

**Auxiliary Losses** [[PROPOSED - Not yet implemented]]:

1. **Structure Reconstruction Loss** (EGNN regularization):
   $$
   \mathcal{L}_{\text{recon}} = \|\mathbf{x}_{\text{pred}} - \mathbf{x}_{\text{true}}\|^2
   $$
   Where $\mathbf{x}_{\text{pred}}$ are refined coordinates from EGNN.
   
   **Purpose:** Encourage physically realistic coordinate updates (Amber force field alignment)

2. **Contrastive Loss** (Multimodal alignment):
   $$
   \mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_{\text{seq}}, \mathbf{z}_{\text{struct}}) / \tau)}{\sum_{j} \exp(\text{sim}(\mathbf{z}_{\text{seq}}, \mathbf{z}_{\text{struct}}^{(j)}) / \tau)}
   $$
   
   **Purpose:** Align sequence and structure embeddings in shared space (CLIP-style)

3. **pLDDT-Weighted Loss**:
   $$
   \mathcal{L}_{\text{weighted}} = \sum_{i=1}^N w_i (\hat{y}_i - y_i)^2, \quad w_i = \frac{\text{pLDDT}_i}{100}
   $$
   
   **Purpose:** Downweight samples with low-confidence structures

**Total Loss:**
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_1 \mathcal{L}_{\text{recon}} + \lambda_2 \mathcal{L}_{\text{contrast}} + \lambda_{\text{reg}} \|\theta\|^2
$$

Where $\lambda_1 = 0.1$, $\lambda_2 = 0.05$, $\lambda_{\text{reg}} = 1 \times 10^{-4}$.

---

### 8.5 Batch Construction and DataLoader Configuration

**Collate Function (Handles Variable-Size Graphs):**

```python
from torch_geometric.data import Batch as PyGBatch

def collate_multimodal(batch):
    """
    Custom collate function for multimodal batches
    Handles both fixed-size sequences and variable-size graphs
    """
    # Stack sequence tokens (fixed size after padding)
    mhc_input_ids = torch.stack([x['mhc_tokens']['input_ids'].squeeze(0) for x in batch])
    mhc_attention_mask = torch.stack([x['mhc_tokens']['attention_mask'].squeeze(0) for x in batch])
    pep_input_ids = torch.stack([x['pep_tokens']['input_ids'].squeeze(0) for x in batch])
    pep_attention_mask = torch.stack([x['pep_tokens']['attention_mask'].squeeze(0) for x in batch])
    
    # Batch graphs (PyG handles variable sizes)
    mhc_graphs = PyGBatch.from_data_list([x['mhc_graph'] for x in batch])
    pep_graphs = PyGBatch.from_data_list([x['pep_graph'] for x in batch])
    
    # Stack labels and flags
    labels = torch.stack([x['label'] for x in batch])
    has_structure = torch.stack([x['has_structure'] for x in batch])
    
    return {
        'mhc_tokens': {'input_ids': mhc_input_ids, 'attention_mask': mhc_attention_mask},
        'pep_tokens': {'input_ids': pep_input_ids, 'attention_mask': pep_attention_mask},
        'mhc_graph': mhc_graphs,
        'pep_graph': pep_graphs,
        'labels': labels,
        'has_structure': has_structure
    }
```

**DataLoader Configuration:**

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_multimodal,
    drop_last=True  # Ensure consistent batch size
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    collate_fn=collate_multimodal
)
```

---

### 8.6 Training Loop Implementation

**Main Training Function:**

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass
        predictions, fusion_gates = model(
            mhc_tokens=batch['mhc_tokens'],
            pep_tokens=batch['pep_tokens'],
            mhc_graph=batch['mhc_graph'],
            pep_graph=batch['pep_graph'],
            has_structure=batch['has_structure']
        )
        
        # Compute loss
        loss = criterion(predictions, batch['labels'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss
```

**Validation Loop:**

```python
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_gates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            predictions, fusion_gates = model(
                mhc_tokens=batch['mhc_tokens'],
                pep_tokens=batch['pep_tokens'],
                mhc_graph=batch['mhc_graph'],
                pep_graph=batch['pep_graph'],
                has_structure=batch['has_structure']
            )
            
            loss = criterion(predictions, batch['labels'])
            total_loss += loss.item()
            
            all_preds.append(predictions.cpu())
            all_labels.append(batch['labels'].cpu())
            all_gates.append(fusion_gates.cpu())
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    rmse = torch.sqrt(torch.mean((all_preds - all_labels) ** 2))
    r2 = 1 - torch.sum((all_preds - all_labels) ** 2) / torch.sum((all_labels - all_labels.mean()) ** 2)
    
    return total_loss / len(dataloader), rmse.item(), r2.item()
```

---

### 8.7 Checkpoint Management and Model Saving

**Checkpoint Strategy:**

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, patience=10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = float('inf')
        self.patience = patience
        self.patience_counter = 0
    
    def save_checkpoint(self, model, optimizer, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save last checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'last_checkpoint.pt')
        
        # Save best checkpoint
        if metrics['val_rmse'] < self.best_metric:
            self.best_metric = metrics['val_rmse']
            self.patience_counter = 0
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
            print(f"✓ Best model saved (val_rmse: {metrics['val_rmse']:.4f})")
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= self.patience  # Early stopping signal
```

---

### 8.8 Reproducibility and Random Seeds

**Seed Setting:**

```python
def set_all_seeds(seed=42):
    """Ensure reproducibility across runs"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

---

## Summary of Training Setup Section

This section provided **comprehensive training configuration** for all model components:

**Key Specifications:**
1. **Hardware**: A100 40GB GPU, 256GB RAM, Ubuntu 22.04
2. **Software**: PyTorch 2.0.1, PyTorch Geometric 2.3.1, Transformers 4.30.2
3. **Hyperparameters**: Detailed configs for sequence (lr=1e-5), structure (lr=5e-4), fusion (lr=1e-4)
4. **Training Strategy**: Phased training (freeze → unfreeze), mixed precision, gradient clipping
5. **Data Pipeline**: Custom collate function, augmentation strategies, missing structure handling

**Complete Configuration Files:**
- ✅ `sequence/config.yaml` (existing)
- ✅ `structure/config_egnn.yaml` (specified)
- ✅ `multimodal/config_fusion.yaml` (specified)

The next section (Part 7) will detail **evaluation metrics and protocols** for assessing model performance.

---

**End of Part 6/8**  
*Continue to Part 7/8: Evaluation Metrics and Statistical Analysis*
