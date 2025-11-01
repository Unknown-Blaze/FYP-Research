"""
ESM-2 Fine-Tuning Script with Advanced Cross-Attention (v7.2 - WARM START)

KEY IMPROVEMENTS for faster convergence:
    - 🔥 Warm-start initialization: Gates start at 0.1 (90% ESM, 10% attention)
    - 🔥 Separate learning rates: Higher LR for attention, lower for ESM
    - 🔥 Attention pooling initialized to uniform weights
    - 🔥 Progressive unfreezing option
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
import warnings
import os
import sys
import random
import logging
import joblib
import yaml
from functools import partial
from typing import List, Dict, Tuple, Optional, Any, Union
from pydantic import BaseModel, ValidationError, FilePath, DirectoryPath, conint, confloat

warnings.filterwarnings('ignore')


# ===========================================================================
# 0. CONFIGURATION VALIDATION (PYDANTIC)
# ===========================================================================

class AppConfig(BaseModel):
    """Pydantic model to validate the config.yaml file."""
    # --- Core Paths ---
    data_file: FilePath
    output_dir: str

    # --- Model ---
    model_name: str
    tokenizer_name: str
    max_length: conint(gt=0) = 512
    
    # --- Cross-Attention Config ---
    freeze_esm: bool = False
    num_ca_layers: conint(gt=0) = 2
    num_attention_heads: conint(gt=0) = 8
    ca_dropout: confloat(ge=0, lt=1) = 0.1
    
    # NEW: Warm start configuration
    warm_start_gate: confloat(ge=0, le=1) = 0.1  # Start with 10% attention, 90% ESM

    # --- Data Columns ---
    mhc_col: str
    pep_col: str
    label_col: str

    # --- Data Preprocessing ---
    convert_to_pic50: bool
    normalize: bool

    # --- Data Splitting & Reproducibility ---
    seed: int
    test_size: confloat(gt=0, lt=1)
    valid_size: confloat(gt=0, lt=1)

    # --- Training Hyperparameters ---
    epochs: conint(gt=0)
    batch_size: conint(gt=0)
    accum_steps: conint(gt=0)
    lr: confloat(gt=0)
    lr_attention_multiplier: confloat(gt=0) = 10.0  # NEW: Higher LR for attention
    scheduler_patience: conint(ge=0)
    clip_norm: confloat(gt=0)
    early_stopping_patience: conint(gt=0)

    # --- Hardware & Performance ---
    use_amp: bool
    num_workers: conint(ge=0)


# ===========================================================================
# 1. LOGGING SETUP
# ===========================================================================

def setup_logging(output_dir: str) -> None:
    """Configures the logging module."""
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_path = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


# ===========================================================================
# 2. REPRODUCIBILITY
# ===========================================================================

def set_seed(seed_value: int) -> None:
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed_value}")


# ===========================================================================
# 3. ADVANCED CROSS-ATTENTION MODEL WITH WARM START
# ===========================================================================

class GatedFusion(nn.Module):
    """
    Gated fusion with warm-start initialization.
    Starts favoring original ESM features (gate ≈ 0.1) to preserve pretrained knowledge.
    """
    def __init__(self, embed_dim: int, initial_gate: float = 0.1):
        super().__init__()
        self.gate_transform = nn.Linear(embed_dim * 2, embed_dim)
        
        # WARM START: Initialize to produce small gate values initially
        # This means: output ≈ 0.1 * cross_attended + 0.9 * original
        with torch.no_grad():
            # Bias to produce negative values → sigmoid → small values
            self.gate_transform.bias.fill_(np.log(initial_gate / (1 - initial_gate)))
            # Small weights to start
            self.gate_transform.weight.normal_(0, 0.01)
        
        logging.info(f"  ✓ Initialized GatedFusion with warm start (gate ≈ {initial_gate:.2f})")
        
    def forward(self, original: torch.Tensor, cross_attended: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([original, cross_attended], dim=-1)
        gate = torch.sigmoid(self.gate_transform(combined))
        return gate * cross_attended + (1 - gate) * original


class CoAttentionLayer(nn.Module):
    """
    Co-attention layer with warm-start initialization.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, initial_gate: float = 0.1):
        super().__init__()
        
        self.mhc_to_pep_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pep_to_mhc_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.mhc_fusion = GatedFusion(embed_dim, initial_gate)
        self.pep_fusion = GatedFusion(embed_dim, initial_gate)
        
        self.mhc_norm = nn.LayerNorm(embed_dim)
        self.pep_norm = nn.LayerNorm(embed_dim)
        
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
        
        # Initialize FFN with smaller weights for stability
        for module in [self.mhc_ffn, self.pep_ffn]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
    def forward(
        self, 
        mhc_hidden: torch.Tensor, 
        pep_hidden: torch.Tensor,
        mhc_mask: torch.Tensor,
        pep_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # MHC attends to peptide
        mhc_attended, _ = self.mhc_to_pep_attn(
            query=mhc_hidden,
            key=pep_hidden,
            value=pep_hidden,
            key_padding_mask=~pep_mask
        )
        mhc_fused = self.mhc_fusion(mhc_hidden, mhc_attended)
        mhc_normed = self.mhc_norm(mhc_fused)
        
        # Peptide attends to MHC
        pep_attended, _ = self.pep_to_mhc_attn(
            query=pep_hidden,
            key=mhc_hidden,
            value=mhc_hidden,
            key_padding_mask=~mhc_mask
        )
        pep_fused = self.pep_fusion(pep_hidden, pep_attended)
        pep_normed = self.pep_norm(pep_fused)
        
        # Feed-forward with residual
        mhc_out = mhc_normed + self.mhc_ffn(mhc_normed)
        pep_out = pep_normed + self.pep_ffn(pep_normed)
        
        return mhc_out, pep_out


class AttentionPooling(nn.Module):
    """
    Attention pooling with uniform initialization.
    Starts similar to mean pooling, gradually learns importance.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Initialize to produce approximately uniform attention
        with torch.no_grad():
            for module in self.attention:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        logging.info("  ✓ Initialized AttentionPooling with near-uniform weights")
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attention_scores = self.attention(hidden_states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        pooled = (hidden_states * attention_weights).sum(dim=1)
        return pooled


class InteractionModule(nn.Module):
    """
    Explicit interaction modeling with proper initialization.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, mhc_pooled: torch.Tensor, pep_pooled: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([mhc_pooled, pep_pooled], dim=-1)
        product = mhc_pooled * pep_pooled
        difference = torch.abs(mhc_pooled - pep_pooled)
        combined = torch.cat([concat, product, difference], dim=-1)
        return self.projection(combined)


class ESM2FineTuneModel(nn.Module):
    """
    Advanced ESM-2 model with warm-start initialization for faster convergence.
    """
    def __init__(
        self, 
        esm_model_name: str, 
        tokenizer_name: str, 
        num_ca_layers: int = 2,
        num_heads: int = 8,
        hidden_dim: int = 512, 
        dropout: float = 0.2,
        freeze_esm: bool = False,
        warm_start_gate: float = 0.1
    ):
        super().__init__()
        logging.info(f"Loading model: {esm_model_name}...")
        logging.info(f"Configuration:")
        logging.info(f"  - Co-attention layers: {num_ca_layers}")
        logging.info(f"  - Attention heads: {num_heads}")
        logging.info(f"  - ESM frozen: {freeze_esm}")
        logging.info(f"  - Warm start gate: {warm_start_gate}")
        
        try:
            self.esm_model = AutoModel.from_pretrained(esm_model_name)
        except:
            # ESMplusplus might need special loading
            self.esm_model = AutoModel.from_pretrained(esm_model_name, trust_remote_code=True)

        embed_dim = self.esm_model.config.hidden_size
        
        logging.info(f"Model type: {type(self.esm_model).__name__}")
        logging.info(f"Embedding dimension: {embed_dim}")
        
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
            logging.info("✓ ESM encoder frozen")
        
        self.co_attention_layers = nn.ModuleList([
            CoAttentionLayer(embed_dim, num_heads, dropout, warm_start_gate)
            for _ in range(num_ca_layers)
        ])
        
        self.mhc_pooling = AttentionPooling(embed_dim)
        self.pep_pooling = AttentionPooling(embed_dim)
        
        self.interaction_module = InteractionModule(embed_dim, dropout)
        
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

    def forward(
        self, 
        mhc_batch: Dict[str, torch.Tensor], 
        pep_batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        device = next(self.esm_model.parameters()).device
        mhc_batch = {k: v.to(device) for k, v in mhc_batch.items()}
        pep_batch = {k: v.to(device) for k, v in pep_batch.items()}
        
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.esm_model.parameters())):
            mhc_hidden = self.esm_model(**mhc_batch).last_hidden_state
            pep_hidden = self.esm_model(**pep_batch).last_hidden_state
        
        mhc_mask = mhc_batch['attention_mask'].bool()
        pep_mask = pep_batch['attention_mask'].bool()
        
        mhc_original = mhc_hidden
        pep_original = pep_hidden
        
        for co_attn_layer in self.co_attention_layers:
            mhc_hidden, pep_hidden = co_attn_layer(
                mhc_hidden, pep_hidden, mhc_mask, pep_mask
            )
        
        mhc_orig_pooled = self.mhc_pooling(mhc_original, mhc_mask)
        pep_orig_pooled = self.pep_pooling(pep_original, pep_mask)
        mhc_ca_pooled = self.mhc_pooling(mhc_hidden, mhc_mask)
        pep_ca_pooled = self.pep_pooling(pep_hidden, pep_mask)
        
        interaction_features = self.interaction_module(mhc_ca_pooled, pep_ca_pooled)
        
        combined = torch.cat([
            mhc_orig_pooled,
            pep_orig_pooled,
            mhc_ca_pooled,
            pep_ca_pooled,
            interaction_features
        ], dim=1)
        
        return self.regression_head(combined)


# ===========================================================================
# 4-6. DATASET, COLLATE, EVALUATION, DATA PREP (Same as before)
# ===========================================================================

class MHCSequenceDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer: PreTrainedTokenizer, 
        mhc_col: str, 
        peptide_col: str, 
        ic50_col: str, 
        max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        try:
            self.mhc_seqs: List[str] = df[mhc_col].tolist()
            self.pep_seqs: List[str] = df[peptide_col].tolist()
            self.labels: np.ndarray = df[ic50_col].values.astype(np.float32)
        except KeyError as e:
            logging.fatal(f"FATAL ERROR: Column not found in CSV: {e}")
            sys.exit(1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mhc_seq: str = self.mhc_seqs[idx]
        pep_seq: str = self.pep_seqs[idx]
        label: np.float32 = self.labels[idx]

        mhc_tokenized = self.tokenizer(
            mhc_seq, return_tensors="pt", padding=False, 
            truncation=True, max_length=self.max_length
        )
        pep_tokenized = self.tokenizer(
            pep_seq, return_tensors="pt", padding=False, 
            truncation=True, max_length=self.max_length
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
            'label': torch.tensor(label, dtype=torch.float32)
        }


def collate_fn(
    batch_list: List[Dict[str, Any]], 
    pad_token_id: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    labels = torch.stack([item['label'] for item in batch_list])
    
    mhc_batches: Dict[str, torch.Tensor] = {}
    pep_batches: Dict[str, torch.Tensor] = {}
    
    mhc_batches['input_ids'] = pad_sequence(
        [item['mhc_batch']['input_ids'] for item in batch_list],
        batch_first=True, padding_value=pad_token_id
    )
    pep_batches['input_ids'] = pad_sequence(
        [item['pep_batch']['input_ids'] for item in batch_list],
        batch_first=True, padding_value=pad_token_id
    )
    
    mhc_batches['attention_mask'] = pad_sequence(
        [item['mhc_batch']['attention_mask'] for item in batch_list],
        batch_first=True, padding_value=0
    )
    pep_batches['attention_mask'] = pad_sequence(
        [item['pep_batch']['attention_mask'] for item in batch_list],
        batch_first=True, padding_value=0
    )
    
    return mhc_batches, pep_batches, labels


def evaluate(
    model: nn.Module, 
    data_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    
    model.eval()
    total_loss: float = 0.0
    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for batch in data_loader:
            mhc_batch, pep_batch, labels = batch
            labels = labels.to(device)
            predictions = model(mhc_batch, pep_batch)
            loss = criterion(predictions.squeeze(), labels)
            total_loss += loss.item()
            all_preds.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    
    all_preds_np = np.array(all_preds).reshape(-1, 1)
    all_targets_np = np.array(all_targets).reshape(-1, 1)
    all_preds_orig: np.ndarray = all_preds_np
    all_targets_orig: np.ndarray = all_targets_np

    if scaler:
        try:
            all_preds_orig = scaler.inverse_transform(all_preds_np)
            all_targets_orig = scaler.inverse_transform(all_targets_np)
        except ValueError as e:
            logging.warning(f"Could not inverse-transform. Error: {e}")
    
    rmse = np.sqrt(mean_squared_error(all_targets_orig, all_preds_orig))
    r2 = r2_score(all_targets_orig, all_preds_orig)
    pcc, _ = pearsonr(all_targets_orig.flatten(), all_preds_orig.flatten())
    scc, _ = spearmanr(all_targets_orig.flatten(), all_preds_orig.flatten())

    return avg_loss, float(rmse), float(r2), float(pcc), float(scc), \
           all_preds_orig.flatten(), all_targets_orig.flatten()


def prepare_data(config: AppConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    """Loads, cleans, splits, and preprocesses data."""
    logging.info("Loading and splitting data...")
    try:
        all_df: pd.DataFrame = pd.read_csv(config.data_file, on_bad_lines='skip')
        
        initial_count = len(all_df)
        all_df = all_df.dropna(subset=[config.mhc_col, config.pep_col, config.label_col])
        all_df = all_df[all_df[config.pep_col].str.isalpha()]
        all_df = all_df[all_df[config.mhc_col].str.isalpha()]
        clean_count = len(all_df)
        logging.info(f"Loaded {initial_count} data points, {clean_count} remaining after cleaning.")
        
        train_val_df, test_df = train_test_split(
            all_df, test_size=config.test_size, random_state=config.seed
        )
        
        valid_size_scaled = config.valid_size / (1.0 - config.test_size)
        train_df, valid_df = train_test_split(
            train_val_df, test_size=valid_size_scaled, random_state=config.seed
        )

        logging.info(f"  Training: {len(train_df)}, Validation: {len(valid_df)}, Test: {len(test_df)}")

    except FileNotFoundError:
        logging.fatal(f"FATAL ERROR: File not found: {config.data_file}")
        sys.exit(1)

    if config.convert_to_pic50:
        logging.info("Converting to pIC50...")
        for df in [train_df, valid_df, test_df]:
            df[config.label_col] = df[config.label_col].apply(lambda x: min(float(x), 50000.0))
            molar_values = df[config.label_col] * 1e-9
            df[config.label_col] = -np.log10(molar_values + 1e-12)

    label_scaler: Optional[MinMaxScaler] = None
    
    if config.normalize:
        logging.info("Normalizing labels...")
        label_scaler = MinMaxScaler()
        train_labels = train_df[config.label_col].values.reshape(-1, 1)
        label_scaler.fit(train_labels)
        
        scaler_path = os.path.join(config.output_dir, "label_scaler.joblib")
        joblib.dump(label_scaler, scaler_path)
        
        train_df[config.label_col] = label_scaler.transform(train_labels)
        valid_df[config.label_col] = label_scaler.transform(valid_df[config.label_col].values.reshape(-1, 1))
        test_df[config.label_col] = label_scaler.transform(test_df[config.label_col].values.reshape(-1, 1))
            
    return train_df, valid_df, test_df, label_scaler


# ===========================================================================
# 7. TRAINING WITH SEPARATE LEARNING RATES
# ===========================================================================

def run_training(config: AppConfig, device: torch.device) -> None:
    """Training loop with separate LRs for ESM and attention layers."""
    
    log_dir = os.path.join(config.output_dir, 'runs')
    writer = SummaryWriter(log_dir=log_dir)
    
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    last_checkpoint_path = os.path.join(config.output_dir, "last_checkpoint.pt")
    
    train_df, valid_df, test_df, label_scaler = prepare_data(config)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    train_dataset = MHCSequenceDataset(train_df, tokenizer, config.mhc_col, config.pep_col, config.label_col, config.max_length)
    valid_dataset = MHCSequenceDataset(valid_df, tokenizer, config.mhc_col, config.pep_col, config.label_col, config.max_length)
    test_dataset = MHCSequenceDataset(test_df, tokenizer, config.mhc_col, config.pep_col, config.label_col, config.max_length)

    collate_wrapper = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_wrapper)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_wrapper)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_wrapper)

    model: nn.Module = ESM2FineTuneModel(
        esm_model_name=config.model_name,
        tokenizer_name=config.tokenizer_name,
        num_ca_layers=config.num_ca_layers,
        num_heads=config.num_attention_heads,
        dropout=config.ca_dropout,
        freeze_esm=config.freeze_esm,
        warm_start_gate=config.warm_start_gate
    ).to(device)
    
    criterion: nn.Module = nn.MSELoss()
    
    # Separate parameter groups with different learning rates
    esm_params = []
    attention_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'esm_model' in name:
            esm_params.append(param)
        else:
            attention_params.append(param)
    
    param_groups = []
    if esm_params:
        param_groups.append({'params': esm_params, 'lr': config.lr})
        logging.info(f"ESM parameters: {sum(p.numel() for p in esm_params):,} (lr={config.lr})")
    if attention_params:
        attention_lr = config.lr * config.lr_attention_multiplier
        param_groups.append({'params': attention_params, 'lr': attention_lr})
        logging.info(f"Attention parameters: {sum(p.numel() for p in attention_params):,} (lr={attention_lr})")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.scheduler_patience, factor=0.1)

    best_valid_rmse: float = float('inf')
    best_epoch: int = 0
    start_epoch: int = 0
    epochs_no_improve: int = 0
    
    logging.info("\n" + "=" * 70)
    logging.info("Starting training with WARM-START CROSS-ATTENTION...")
    logging.info(f"  Batch Size: {config.batch_size}")
    logging.info(f"  ESM LR: {config.lr}")
    logging.info(f"  Attention LR: {config.lr * config.lr_attention_multiplier}")
    logging.info("-" * 70)

    scaler = GradScaler(enabled=config.use_amp)
    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss: float = 0.0
        optimizer.zero_grad()

        for i, (mhc_batch, pep_batch, labels) in enumerate(train_loader):
            labels = labels.to(device)
            with autocast(enabled=config.use_amp):
                predictions = model(mhc_batch, pep_batch)
                loss = criterion(predictions.squeeze(), labels)
            
            loss = loss / config.accum_steps
            train_loss += loss.item() * config.accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % config.accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        
        valid_loss, valid_rmse, valid_r2, valid_pcc, valid_scc, _, _ = evaluate(
            model, valid_loader, criterion, device, scaler=label_scaler
        )
        
        logging.info(f"\nEpoch {epoch+1}/{config.epochs}")
        logging.info(f"  Train Loss: {avg_train_loss:.4f}")
        logging.info(f"  Valid RMSE: {valid_rmse:.4f}, R²: {valid_r2:.4f}, PCC: {valid_pcc:.4f}")
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Metrics/valid_rmse', valid_rmse, epoch)
        writer.add_scalar('Metrics/valid_r2', valid_r2, epoch)
        
        scheduler.step(valid_loss)
        
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  ✓ Best model saved (RMSE: {valid_rmse:.4f})")
        else:
            epochs_no_improve += 1
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_valid_rmse': best_valid_rmse
        }, last_checkpoint_path)
        
        if epochs_no_improve >= config.early_stopping_patience:
            logging.info(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Testing
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_rmse, test_r2, test_pcc, test_scc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler=label_scaler
    )
    
    logging.info(f"\n{'='*70}")
    logging.info(f"TEST RESULTS:")
    logging.info(f"  RMSE: {test_rmse:.4f}")
    logging.info(f"  R²: {test_r2:.4f}")
    logging.info(f"  PCC: {test_pcc:.4f}")
    logging.info(f"{'='*70}")
    
    pred_df = pd.DataFrame({'true': test_targets, 'pred': test_preds})
    pred_df.to_csv(os.path.join(config.output_dir, "predictions.csv"), index=False)
    
    writer.close()


# ===========================================================================
# 8. MAIN
# ===========================================================================

def load_config(config_path: str) -> AppConfig:
    """Loads and validates config."""
    try:
        with open(config_path, 'r') as f:
            config_data: Dict[str, Any] = yaml.safe_load(f)
        
        config = AppConfig(**config_data)
        os.makedirs(config.output_dir, exist_ok=True)
        return config
        
    except FileNotFoundError:
        print(f"FATAL: Config not found: {config_path}")
        sys.exit(1)
    except ValidationError as e:
        print(f"FATAL: Config error:\n{e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ESM-2 Warm-Start Cross-Attention (v7.2)")
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    setup_logging(config.output_dir)
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    run_training(config, device)


if __name__ == "__main__":
    main()
