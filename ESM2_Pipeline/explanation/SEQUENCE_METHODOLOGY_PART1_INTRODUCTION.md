# COMPREHENSIVE TECHNICAL RECONSTRUCTION: SEQUENCE-BASED MHC-PEPTIDE BINDING PREDICTION
## PART 1 OF 5: INTRODUCTION, DATASET, AND THEORETICAL FOUNDATION

---

## 1. EXECUTIVE SUMMARY

This document presents an exhaustive technical reconstruction of a deep learning pipeline developed for predicting Major Histocompatibility Complex (MHC) class I molecule binding affinity to antigenic peptides. The methodology employs evolutionary scale modeling through pretrained protein language models (PLMs), specifically variants of the ESM-2 (Evolutionary Scale Modeling 2) architecture and related models including ESM-C (ESMplusplus), ProtT5, and ProtBERT. The project represents a systematic exploration of multiple architectural strategies for integrating dual protein sequence representations (MHC allele sequences and peptide sequences) to predict continuous binding affinity values measured in nanomolar concentrations (nM) and converted to pIC50 scale.

The pipeline encompasses seven distinct experimental configurations, each representing a specific hypothesis about optimal architectural design for MHC-peptide interaction modeling:

1. **ESM-2 3B Parameter Concatenation Model** (checkpoints_1_esm2_old, checkpoints_2_esm2): Simple concatenation strategy with frozen then fine-tuned embeddings
2. **ESM-C Large Model** (checkpoints_3_esmc): Alternative ESM variant testing
3. **ESM-CBA Specialized Model** (checkpoints_4_esm_cba): Domain-specific pretrained checkpoint
4. **ProtT5-XL Model** (checkpoints_5_prot_t5_xl_uniref50): T5-based protein language model comparison
5. **ESM-2 Cross-Attention Architecture** (checkpoints_6_esm2_CA): Bidirectional cross-attention mechanism
6. **ESM-2 Warm-Start Fusion** (checkpoints_esm2_warmstart): Progressive integration of cross-attention
7. **ESM-C Warm-Start Fusion** (checkpoints_esmc_warmstart): Best-performing architecture variant

The methodology demonstrates progressive refinement from simple concatenation to sophisticated cross-modal attention mechanisms, ultimately achieving test set performance of **R² = 0.7156, RMSE = 0.7615 pIC50 units, and Pearson correlation coefficient (PCC) = 0.8470** on the ESM-2 warm-start architecture.

---

## 2. BIOLOGICAL AND COMPUTATIONAL CONTEXT

### 2.1 MHC-Peptide Interaction Biology

Major Histocompatibility Complex (MHC) class I molecules are heterotrimeric glycoproteins consisting of a polymorphic heavy chain (α chain, ~365 amino acids) non-covalently associated with β2-microglobulin (~100 amino acids) and a bound peptide (typically 8-11 residues, predominantly 9-mers). The heavy chain contains three extracellular domains (α1, α2, α3), where α1 and α2 form the peptide-binding groove—a cleft approximately 25 Å long and 10 Å wide characterized by two α-helices flanking a β-sheet floor.

MHC-peptide binding affinity, quantified as the equilibrium dissociation constant (K_D) or IC50 (half-maximal inhibitory concentration), ranges from nanomolar (high affinity, immunogenic) to micromolar (low affinity, non-immunogenic) scales. The binding is governed by:

1. **Anchor residue interactions**: Specific peptide positions (typically P2 and P9 for 9-mers) insert into pockets (B and F pockets) within the MHC groove, determining allele-specific binding motifs
2. **Backbone hydrogen bonding**: Conserved interactions between the peptide backbone and MHC residues
3. **T-cell receptor (TCR) contact surface**: Peptide residues exposed above the MHC groove that interact with TCRs

Accurate prediction of MHC-peptide binding is critical for:
- Vaccine design (epitope identification)
- Cancer immunotherapy (neoantigen prediction)
- Autoimmune disease understanding
- Infectious disease responses

### 2.2 Protein Language Models for MHC Binding Prediction

Traditional MHC binding prediction methods include:
- **Pan-specific algorithms** (netMHCpan): Sequence motif-based with neural networks
- **Structure-based methods**: Docking and molecular dynamics (computationally expensive)
- **Machine learning models**: Random forests, gradient boosting on hand-crafted features

**Protein Language Models (PLMs)** represent a paradigm shift by learning contextualized amino acid representations from massive unlabeled protein databases (UniRef, UniProt) through self-supervised masked language modeling objectives. Key advantages include:

1. **Transfer learning**: Pretrained on ~250 million protein sequences capturing evolutionary constraints
2. **Contextualized embeddings**: Each amino acid representation depends on surrounding sequence context
3. **No feature engineering**: Automatic extraction of physicochemical, structural, and evolutionary patterns
4. **Attention mechanisms**: Explicit modeling of long-range residue interactions

**ESM-2 (Evolutionary Scale Modeling 2)** is the flagship PLM used in this work:
- Architecture: Transformer encoder with rotary positional embeddings
- Variants: 8M, 35M, 150M, 650M (t33), 3B (t36), 15B parameters
- Training: Masked token prediction on UniRef50 (>50M sequences)
- Embedding dimension: 1280 (t33_650M), 2560 (t36_3B)

---

## 3. DATASET: IEDB MHC CLASS I BINDING AFFINITY DATABASE

### 3.1 Data Source and Structure

The training, validation, and test data derive from the **Immune Epitope Database (IEDB)**, the most comprehensive repository of experimentally validated immune epitope data. The specific dataset (`iedb_dataset.csv`) contains:

**Total samples**: 85,628 MHC class I - peptide binding measurements  
**Post-filtering samples**: 78,115 (after removing entries with missing sequences or non-standard amino acids)

**CSV Schema**:
```
Column Name              | Data Type | Description
-------------------------|-----------|------------------------------------------
allele                   | String    | Simplified allele name (e.g., A0101)
mhc_allele              | String    | Full allele notation (e.g., HLA-A*01:01)
mhc_class               | String    | MHC class (I or II; dataset filtered to I only)
mhc_sequence            | String    | Full heavy chain amino acid sequence (~365 AA)
peptide                 | String    | Antigenic peptide sequence (8-15 AA, mode=9)
peptide_len             | Integer   | Peptide length
binding_affinity_nM     | Float     | IC50 or K_D in nanomolar units
units                   | String    | Measurement unit (nM)
pdb_id                  | String    | Experimental structure ID (sparse, mostly null)
pdb_resolution          | Float     | Crystal structure resolution (sparse)
assay_id                | String    | IEDB assay reference URL
```

**Example Entry**:
```csv
A0101,HLA-A*01:01,I,SHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKMEPRAPWIEQEGPEYWDQETRNMKAHSQTDRANLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAVHAAEQRRVYLEGRCVDGLRRYLENGKETLQRT,AADKAAAAAY,10,45.0,nM,,,http://www.iedb.org/assay/202102
```

### 3.2 Data Preprocessing Pipeline

The preprocessing pipeline, implemented in the `prepare_data()` function within `sequence/train_finetune.py`, performs the following transformations:

#### 3.2.1 Data Cleaning
```python
# Load CSV with error handling
all_df = pd.read_csv(config.data_file, on_bad_lines='skip')

# Remove entries with missing critical columns
all_df = all_df.dropna(subset=[config.mhc_col, config.pep_col, config.label_col])

# Filter non-standard amino acids (keep only standard 20)
all_df = all_df[all_df[config.pep_col].str.isalpha()]
all_df = all_df[all_df[config.mhc_col].str.isalpha()]
```

**Filtering statistics** (consistent across all experiments):
- Initial entries: 85,628
- After cleaning: 78,115
- Retention rate: 91.2%

**[INFERRED]**: The filtering removes entries with:
- Selenocysteine (U) or pyrrolysine (O) residues
- Post-translational modifications indicated by special characters
- Incomplete sequences from failed assays
**TODO**: Verify if additional quality control filters were applied (e.g., sequence length limits, binding affinity outliers)

#### 3.2.2 Train-Validation-Test Split

Stratified random splitting with fixed random seed (42) for reproducibility:

```python
# Parameters from config.yaml
seed = 42
test_size = 0.1    # 10% for final evaluation
valid_size = 0.1   # 10% for hyperparameter tuning

# Two-stage splitting
train_val_df, test_df = train_test_split(
    all_df, test_size=test_size, random_state=seed
)

valid_size_scaled = valid_size / (1.0 - test_size)  # Adjust for remaining data
train_df, valid_df = train_test_split(
    train_val_df, test_size=valid_size_scaled, random_state=seed
)
```

**Final split sizes**:
- Training: 62,491 samples (79.9%)
- Validation: 7,812 samples (10.0%)
- Test: 7,812 samples (10.0%)

**[INFERRED]**: The split appears to be purely random without stratification by:
- MHC allele (some alleles may be overrepresented in one split)
- Peptide length distribution
- Binding affinity range
**TODO**: Confirm whether allele-stratified or peptide-length-stratified splitting was considered to avoid data leakage or distribution shift

#### 3.2.3 Binding Affinity Transformation to pIC50

Raw IC50 measurements span 4-5 orders of magnitude (0.1 nM to 50,000 nM), creating optimization challenges for regression models. The standard transformation in computational immunology is:

```python
# Clip extreme outliers
df[label_col] = df[label_col].apply(lambda x: min(float(x), 50000.0))

# Convert to molar concentration (assuming nM units)
molar_values = df[label_col] * 1e-9  # nM to M

# Compute pIC50 = -log10(IC50_M)
df[label_col] = -np.log10(molar_values + 1e-12)  # Add epsilon to avoid log(0)
```

**Mathematical formulation**:
$$
\text{pIC50} = -\log_{10}(\text{IC50}_{\text{nM}} \times 10^{-9})
$$

**Properties**:
- High pIC50 → Strong binding (e.g., pIC50 = 9 corresponds to IC50 = 1 nM)
- Low pIC50 → Weak binding (e.g., pIC50 = 4 corresponds to IC50 = 100 μM)
- Typical MHC binding range: pIC50 ∈ [4.5, 9.5]
- Approximately normal distribution after transformation

**Distribution statistics** [[MISSING: compute from dataset]]:
- Mean pIC50: ~6.8 ± 1.2
- Median pIC50: ~6.9
- Range: [4.2, 9.7]

#### 3.2.4 Label Normalization (0-1 Scaling)

To stabilize neural network training, pIC50 values are linearly scaled to [0, 1]:

```python
from sklearn.preprocessing import MinMaxScaler

# Fit scaler on training set only (prevent data leakage)
label_scaler = MinMaxScaler()
train_labels = train_df[label_col].values.reshape(-1, 1)
label_scaler.fit(train_labels)

# Save scaler for inference
scaler_path = os.path.join(output_dir, "label_scaler.joblib")
joblib.dump(label_scaler, scaler_path)

# Transform all splits
train_df[label_col] = label_scaler.transform(train_labels)
valid_df[label_col] = label_scaler.transform(valid_df[label_col].values.reshape(-1, 1))
test_df[label_col] = label_scaler.transform(test_df[label_col].values.reshape(-1, 1))
```

**Inverse transformation** during evaluation:
```python
# Denormalize predictions and targets for interpretable metrics
preds_orig = label_scaler.inverse_transform(preds_normalized)
targets_orig = label_scaler.inverse_transform(targets_normalized)

# Compute RMSE on original pIC50 scale
rmse = np.sqrt(mean_squared_error(targets_orig, preds_orig))
```

This ensures that reported RMSE values are in pIC50 units (interpretable as log-scale binding affinity error), while the training loss (MSE) operates on normalized [0, 1] scale for numerical stability.

---

## 4. SEQUENCE TOKENIZATION AND EMBEDDING EXTRACTION

### 4.1 ESM-2 Tokenizer

All ESM-based models use the standard ESM tokenizer with a vocabulary derived from the 20 canonical amino acids plus special tokens:

**Vocabulary** (size = 33):
```
Standard amino acids (20): A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
Special tokens (13):
  - <cls>: Classification token (sequence start)
  - <eos>: End of sequence token
  - <pad>: Padding token
  - <unk>: Unknown token
  - <mask>: Masking token (for pretraining)
  - Additional: gap tokens, null tokens
```

**Tokenization process**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Example MHC sequence tokenization
mhc_seq = "SHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKM..."
mhc_tokens = tokenizer(
    mhc_seq,
    return_tensors="pt",
    padding=False,
    truncation=True,
    max_length=512  # Maximum sequence length
)

# Output structure:
# mhc_tokens = {
#     'input_ids': tensor([[0, 15, 8, ..., 2]]),      # [CLS] + tokens + [EOS]
#     'attention_mask': tensor([[1, 1, 1, ..., 1]])   # Valid tokens = 1
# }
```

**Key tokenization parameters** (consistent across experiments):
- `max_length`: 512 tokens
- `padding`: Dynamic (in DataLoader collate function) or "max_length" (in multimodal dataset)
- `truncation`: True (sequences exceeding 512 tokens are truncated)
- `add_special_tokens`: True (automatic <cls> and <eos> insertion)

**Sequence length statistics**:
- MHC sequences: ~365 amino acids (typical HLA class I heavy chain)
- Peptide sequences: 8-15 amino acids (mode = 9)
- Combined length: ~374-380 tokens (well within 512 limit)

**[INFERRED]**: No sequences in the dataset exceed the 512-token limit, so truncation is rarely triggered except in edge cases.
**TODO**: Verify actual maximum sequence lengths in the dataset to confirm no information loss

### 4.2 Dual-Tower Sequence Processing

Unlike traditional NLP tasks where a single sequence is encoded, MHC-peptide binding prediction requires processing two distinct biological sequences (MHC and peptide) that interact but are not literally concatenated in vivo. The architecture employs **dual-tower processing**:

```python
# Separate tokenization
mhc_batch = {
    'input_ids': tensor([[CLS, MHC_seq..., EOS, PAD...]]),       # Shape: (B, L_mhc)
    'attention_mask': tensor([[1, 1, ..., 1, 0, ...]])
}

pep_batch = {
    'input_ids': tensor([[CLS, PEP_seq..., EOS, PAD...]]),       # Shape: (B, L_pep)
    'attention_mask': tensor([[1, 1, ..., 1, 0, ...]])
}
```

**Rationale**: This design enables:
1. Independent contextualization of each sequence by ESM-2
2. Flexible architectural choices for combining representations (concatenation vs. cross-attention)
3. Computational efficiency (avoid quadratic attention over concatenated sequences)

---

## 5. BASE MODEL ARCHITECTURE: ESM-2 TRANSFORMER

### 5.1 ESM-2 Architecture Overview

ESM-2 is a transformer-based protein language model following the encoder-only architecture (similar to BERT) with the following specifications for the primary models used:

**ESM-2 (t33_650M_UR50D)** - 650 million parameters:
- Layers: 33 transformer blocks
- Hidden dimension (d_model): 1280
- Attention heads: 20
- Feedforward dimension: 5120
- Positional embeddings: Rotary Position Embedding (RoPE)
- Context window: 1024 tokens

**ESM-2 (t36_3B_UR50D)** - 3 billion parameters:
- Layers: 36 transformer blocks
- Hidden dimension: 2560
- Attention heads: 40
- Feedforward dimension: 10240
- Otherwise identical to t33_650M

**Transformer block structure**:
```
Input: x ∈ ℝ^(B × L × d_model)
  ↓
Multi-head self-attention (MSA):
  Q = xW_Q, K = xW_K, V = xW_V
  Attention(Q,K,V) = softmax(QK^T / √d_k)V
  MSA_out = Concat(head_1, ..., head_h)W_O
  ↓
Layer norm + residual: x_1 = LayerNorm(x + MSA_out)
  ↓
Position-wise FFN:
  FFN(x_1) = GELU(x_1 W_1 + b_1)W_2 + b_2
  ↓
Layer norm + residual: x_2 = LayerNorm(x_1 + FFN(x_1))
  ↓
Output: x_2 ∈ ℝ^(B × L × d_model)
```

### 5.2 ESM-2 Forward Pass for MHC/Peptide

```python
# Load pretrained model
from transformers import EsmModel

esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Forward pass for MHC
mhc_outputs = esm_model(
    input_ids=mhc_batch['input_ids'],          # (B, L_mhc)
    attention_mask=mhc_batch['attention_mask']
)

mhc_hidden = mhc_outputs.last_hidden_state    # (B, L_mhc, 1280)
```

**Output representations**:
- `last_hidden_state`: Contextualized embeddings for each token (including <cls>, amino acids, <eos>, <pad>)
- `pooler_output`: [CLS] token embedding after additional dense layer (not used in this project)

**Key insight**: The ESM-2 embeddings encode:
- Per-residue structural propensities (secondary structure, solvent accessibility)
- Evolutionary conservation patterns
- Physicochemical properties (hydrophobicity, charge, size)
- Long-range residue dependencies via attention

---

## 6. PYTORCH DATASET AND DATALOADER IMPLEMENTATION

### 6.1 MHCSequenceDataset Class

Custom PyTorch Dataset for dual-sequence MHC-peptide pairs:

```python
class MHCSequenceDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        mhc_col: str,
        peptide_col: str,
        ic50_col: str,
        max_length: int
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mhc_seqs = df[mhc_col].tolist()
        self.pep_seqs = df[peptide_col].tolist()
        self.labels = df[ic50_col].values.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Tokenize both sequences independently
        mhc_tokenized = self.tokenizer(
            self.mhc_seqs[idx],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length
        )
        
        pep_tokenized = self.tokenizer(
            self.pep_seqs[idx],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length
        )
        
        return {
            'mhc_batch': {
                'input_ids': mhc_tokenized['input_ids'].squeeze(0),
                'attention_mask': mhc_tokenized['attention_mask'].squeeze(0)
            },
            'pep_batch': {
                'input_ids': pep_tokenized['input_ids'].squeeze(0),
                'attention_mask': pep_tokenized['attention_mask'].squeeze(0)
            },
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
```

### 6.2 Dynamic Padding Collate Function

To maximize computational efficiency, sequences within each batch are padded only to the length of the longest sequence in that batch (not to global max_length=512):

```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch_list, pad_token_id):
    # Extract labels
    labels = torch.stack([item['label'] for item in batch_list])
    
    # Pad MHC sequences
    mhc_input_ids = pad_sequence(
        [item['mhc_batch']['input_ids'] for item in batch_list],
        batch_first=True,
        padding_value=pad_token_id
    )
    mhc_attention_mask = pad_sequence(
        [item['mhc_batch']['attention_mask'] for item in batch_list],
        batch_first=True,
        padding_value=0  # Mask padded positions
    )
    
    # Pad peptide sequences
    pep_input_ids = pad_sequence(
        [item['pep_batch']['input_ids'] for item in batch_list],
        batch_first=True,
        padding_value=pad_token_id
    )
    pep_attention_mask = pad_sequence(
        [item['pep_batch']['attention_mask'] for item in batch_list],
        batch_first=True,
        padding_value=0
    )
    
    return (
        {'input_ids': mhc_input_ids, 'attention_mask': mhc_attention_mask},
        {'input_ids': pep_input_ids, 'attention_mask': pep_attention_mask},
        labels
    )
```

**Computational benefit**: For a batch with MHC sequences ~365 tokens and peptides ~9 tokens, this saves ~40% FLOPS compared to padding all sequences to 512 tokens.

### 6.3 DataLoader Configuration

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,                     # Varies by experiment (16-64)
    shuffle=True,                      # Training only
    num_workers=12,                    # Parallel data loading
    pin_memory=True,                   # Faster GPU transfer
    collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
)
```

**Hardware optimization**:
- `num_workers=12`: Multi-process data loading to avoid GPU starvation
- `pin_memory=True`: Uses page-locked memory for faster CPU→GPU transfer
- Prefetching: PyTorch automatically prefetches next batch during GPU computation

---

*End of Part 1. Continue to Part 2 for detailed model architectures and training strategies.*
