"""
ESM-2 Fine-Tuning Script (v6 - Pydantic Validation)

CHANGE LOG (from v5):
    - 🛡️ Added Pydantic validation for the config.yaml file.
    - 🆕 Defined an `AppConfig(BaseModel)` to parse and type-check all settings.
    - 🆕 `load_config` now uses `AppConfig` and catches `ValidationError`.
    - ⚙️ Moved `scheduler_patience` from hardcoded to the config file.
    - 🆕 New dependency: `pydantic`
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from transformers import EsmTokenizer, EsmModel, PreTrainedTokenizer
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

# 🆕 --- Pydantic & Type Hinting Imports ---
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
    output_dir: str # Can't validate as DirectoryPath yet, as it might not exist

    # --- Model ---
    model_name: str
    max_length: conint(gt=0) = 512

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
    scheduler_patience: conint(ge=0) # 🆕
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
# 3. MODEL DEFINITION
# ===========================================================================

class ESM2FineTuneModel(nn.Module):
    def __init__(self, esm_model_name: str, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        logging.info(f"Loading ESM-2 model: {esm_model_name}...")
        self.esm_model: nn.Module = EsmModel.from_pretrained(esm_model_name)
        
        esm_dim: int = self.esm_model.config.hidden_size
        
        self.regression_head = nn.Sequential(
            nn.Linear(esm_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def pool_embeddings(self, outputs: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return pooled

    def forward(self, mhc_batch: Dict[str, torch.Tensor], pep_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.esm_model.parameters()).device
        
        mhc_batch_gpu = {k: v.to(device) for k, v in mhc_batch.items()}
        pep_batch_gpu = {k: v.to(device) for k, v in pep_batch.items()}
        
        mhc_outputs = self.esm_model(**mhc_batch_gpu)
        pep_outputs = self.esm_model(**pep_batch_gpu)
        
        mhc_emb = self.pool_embeddings(mhc_outputs, mhc_batch_gpu['attention_mask'])
        pep_emb = self.pool_embeddings(pep_outputs, pep_batch_gpu['attention_mask'])
        
        combined = torch.cat([mhc_emb, pep_emb], dim=1)
        return self.regression_head(combined)

# ===========================================================================
# 4. DATASET & COLLATE FUNCTION
# ===========================================================================

class MHCSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, 
                 mhc_col: str, peptide_col: str, ic50_col: str, max_length: int) -> None:
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        try:
            self.mhc_seqs: List[str] = df[mhc_col].tolist()
            self.pep_seqs: List[str] = df[peptide_col].tolist()
            self.labels: np.ndarray = df[ic50_col].values.astype(np.float32)
        except KeyError as e:
            logging.fatal(f"FATAL ERROR: Column not found in CSV: {e}")
            sys.exit(1)
        except Exception as e:
            logging.fatal(f"FATAL ERROR: Could not process DataFrame. Error: {e}")
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

def collate_fn(batch_list: List[Dict[str, Any]], pad_token_id: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
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

# ===========================================================================
# 5. EVALUATION
# ===========================================================================

def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, 
             device: torch.device, scaler: Optional[MinMaxScaler] = None
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
            logging.warning(f"Could not inverse-transform. Metrics on scaled values. Error: {e}")
    
    rmse = np.sqrt(mean_squared_error(all_targets_orig, all_preds_orig))
    r2 = r2_score(all_targets_orig, all_preds_orig)
    pcc, _ = pearsonr(all_targets_orig.flatten(), all_preds_orig.flatten())
    scc, _ = spearmanr(all_targets_orig.flatten(), all_preds_orig.flatten())

    return avg_loss, float(rmse), float(r2), float(pcc), float(scc), \
           all_preds_orig.flatten(), all_targets_orig.flatten()

# ===========================================================================
# 6. DATA PREPARATION (Helper Function)
# ===========================================================================

def prepare_data(config: AppConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[MinMaxScaler]]:
    """Loads, cleans, splits, and preprocesses data from config."""
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
        
        # Scale validation size correctly
        if (1.0 - config.test_size) == 0:
             logging.fatal("test_size cannot be 1.0. Set to a value < 1.")
             sys.exit(1)
        valid_size_scaled = config.valid_size / (1.0 - config.test_size)
        
        train_df, valid_df = train_test_split(
            train_val_df, test_size=valid_size_scaled, random_state=config.seed
        )

        logging.info(f"  Training set size:   {len(train_df)}")
        logging.info(f"  Validation set size: {len(valid_df)}")
        logging.info(f"  Test set size:       {len(test_df)}")

    except FileNotFoundError:
        logging.fatal(f"FATAL ERROR: File not found: {config.data_file}")
        sys.exit(1)
    except Exception as e:
        logging.fatal(f"FATAL ERROR: Could not read or split CSV. Error: {e}")
        sys.exit(1)

    if config.convert_to_pic50:
        logging.info(f"Converting '{config.label_col}' to pIC50...")
        for df in [train_df, valid_df, test_df]:
            if config.label_col not in df.columns:
                logging.fatal(f"FATAL ERROR: --label_col '{config.label_col}' not found.")
                sys.exit(1)
            df[config.label_col] = df[config.label_col].apply(lambda x: min(float(x), 50000.0))
            molar_values = df[config.label_col] * 1e-9
            df[config.label_col] = -np.log10(molar_values + 1e-12)
        logging.info("✓ Conversion to pIC50 complete.")

    label_scaler: Optional[MinMaxScaler] = None
    scaler_path = os.path.join(config.output_dir, "label_scaler.joblib")
    
    if config.normalize:
        logging.info(f"Applying 0-1 normalization on label column: '{config.label_col}'")
        label_scaler = MinMaxScaler()
        try:
            train_labels = train_df[config.label_col].values.reshape(-1, 1)
            label_scaler.fit(train_labels)
            joblib.dump(label_scaler, scaler_path)
            logging.info(f"✓ Saved label scaler to {scaler_path}")
            
            train_df[config.label_col] = label_scaler.transform(train_labels)
            valid_df[config.label_col] = label_scaler.transform(valid_df[config.label_col].values.reshape(-1, 1))
            test_df[config.label_col] = label_scaler.transform(test_df[config.label_col].values.reshape(-1, 1))
            logging.info("✓ Labels normalized.")
        except KeyError:
            logging.fatal(f"FATAL ERROR: --label_col '{config.label_col}' not found.")
            sys.exit(1)
            
    return train_df, valid_df, test_df, label_scaler

# ===========================================================================
# 7. TRAINING FUNCTION
# ===========================================================================

def run_training(config: AppConfig, device: torch.device) -> None:
    """Main training and validation loop."""
    
    log_dir = os.path.join(config.output_dir, 'runs')
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logs will be saved to: {log_dir}")
    
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    last_checkpoint_path = os.path.join(config.output_dir, "last_checkpoint.pt")
    
    train_df, valid_df, test_df, label_scaler = prepare_data(config)

    logging.info(f"Loading tokenizer: {config.model_name}")
    try:
        tokenizer: PreTrainedTokenizer = EsmTokenizer.from_pretrained(config.model_name)
    except Exception as e:
        logging.fatal(f"Could not load tokenizer. Error: {e}")
        sys.exit(1)

    logging.info(f"Using columns: MHC='{config.mhc_col}', Peptide='{config.pep_col}', Label='{config.label_col}'")
    train_dataset = MHCSequenceDataset(train_df, tokenizer, config.mhc_col, config.pep_col, config.label_col, config.max_length)
    valid_dataset = MHCSequenceDataset(valid_df, tokenizer, config.mhc_col, config.pep_col, config.label_col, config.max_length)
    test_dataset = MHCSequenceDataset(test_df, tokenizer, config.mhc_col, config.pep_col, config.label_col, config.max_length)

    collate_wrapper = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    logging.info(f"Using {config.num_workers} workers for data loading.")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_wrapper)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_wrapper)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_wrapper)

    model: nn.Module = ESM2FineTuneModel(esm_model_name=config.model_name).to(device)
    criterion: nn.Module = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # 🆕 --- Scheduler uses config ---
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', 
        patience=config.scheduler_patience, 
        factor=0.1
    )

    best_valid_rmse: float = float('inf')
    best_epoch: int = 0
    start_epoch: int = 0
    epochs_no_improve: int = 0
    
    if os.path.exists(last_checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {last_checkpoint_path}")
        try:
            checkpoint: Dict[str, Any] = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_valid_rmse = checkpoint['best_valid_rmse']
            best_epoch = checkpoint.get('best_epoch', 0)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            logging.info(f"Resumed from epoch {start_epoch}. Best RMSE: {best_valid_rmse:.4f}")
        except Exception as e:
            logging.warning(f"Could not load checkpoint. Starting from scratch. Error: {e}")
            start_epoch, best_epoch, epochs_no_improve = 0, 0, 0
            best_valid_rmse = float('inf')

    effective_batch_size = config.batch_size * config.accum_steps
    logging.info("\n" + "=" * 70)
    logging.info("Starting training...")
    logging.info(f"  Config: {config.model_dump_json(indent=2)}")
    logging.info(f"  Effective Batch Size: {effective_batch_size}")
    logging.info(f"  AMP: {config.use_amp}")
    logging.info(f"  Early Stopping Patience: {config.early_stopping_patience}")
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
        logging.info(f"  Train Loss: {avg_train_loss:.4f} (scaled)")
        logging.info(f"  Valid Loss: {valid_loss:.4f} (scaled), Valid RMSE: {valid_rmse:.4f}, Valid R²: {valid_r2:.4f}")
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Metrics/valid_rmse', valid_rmse, epoch)
        writer.add_scalar('Metrics/valid_r2', valid_r2, epoch)
        writer.add_scalar('Metrics/valid_pcc', valid_pcc, epoch)
        writer.add_scalar('Metrics/valid_scc', valid_scc, epoch)
        writer.add_scalar('Params/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step(valid_loss)
        
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  ✓ Saved new best model (RMSE: {valid_rmse:.4f}) to {best_model_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"  ! No improvement in {epochs_no_improve} epoch(s).")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_valid_rmse': float(best_valid_rmse),
            'best_epoch': best_epoch,
            'epochs_no_improve': epochs_no_improve,
            'config': config.model_dump()
        }
        torch.save(checkpoint, last_checkpoint_path)
        
        if epochs_no_improve >= config.early_stopping_patience:
            logging.info(f"\n--- EARLY STOPPING ---")
            logging.info(f"Validation RMSE did not improve for {config.early_stopping_patience} epochs.")
            logging.info(f"Stopping at epoch {epoch + 1}.")
            break

    logging.info("\n" + "=" * 70)
    logging.info("=== Testing ===")
    
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f"Loaded best model from {best_model_path} (from epoch {best_epoch + 1})")
    except Exception as e:
        logging.fatal(f"Could not load best model for testing. Error: {e}")
        writer.close()
        sys.exit(1)
        
    final_scaler: Optional[MinMaxScaler] = None
    if config.normalize:
        scaler_path = os.path.join(config.output_dir, "label_scaler.joblib")
        try:
            final_scaler = joblib.load(scaler_path)
            logging.info(f"Loaded label scaler from {scaler_path}")
        except Exception as e:
            logging.warning(f"Could not load scaler for final eval. Error: {e}")

    test_loss, test_rmse, test_r2, test_pcc, test_scc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler=final_scaler
    )
    
    logging.info(f"\nTest Results (on original scale):")
    logging.info(f"  Loss: {test_loss:.4f} (scaled)")
    logging.info(f"  RMSE: {test_rmse:.4f}")
    logging.info(f"  R²:   {test_r2:.4f}")
    logging.info(f"  PCC:  {test_pcc:.4f}")
    logging.info(f"  SCC:  {test_scc:.4f}")
    
    hparams: Dict[str, Any] = config.model_dump()
    for key, val in hparams.items():
        if not isinstance(val, (int, float, str, bool, torch.Tensor)):
            hparams[key] = str(val)

    metrics: Dict[str, Union[float, int]] = {
        'hparam/test_rmse': test_rmse, 'hparam/test_r2': test_r2,
        'hparam/test_pcc': test_pcc, 'hparam/test_scc': test_scc,
        'hparam/best_valid_rmse': best_valid_rmse,
        'hparam/best_epoch': best_epoch
    }
    
    try:
        writer.add_hparams(hparams, metrics)
    except Exception as e:
        logging.warning(f"Could not write HParams to TensorBoard. Error: {e}")
    
    pred_df = pd.DataFrame({'true_label': test_targets, 'pred_label': test_preds})
    pred_path = os.path.join(config.output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    logging.info(f"\n✓ Saved test predictions to {pred_path}")
    
    writer.close()
    logging.info("\nTraining complete. Run finished.")

# ===========================================================================
# 8. EVALUATION-ONLY FUNCTION
# ===========================================================================

def run_evaluation(config: AppConfig, device: torch.device) -> None:
    """Loads a model and runs evaluation on the test set."""
    logging.info("=" * 70)
    logging.info("=== EVALUATION-ONLY MODE ===")
    logging.info("=" * 70)
    
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    scaler_path = os.path.join(config.output_dir, "label_scaler.joblib")
    
    if not os.path.exists(best_model_path):
        logging.fatal(f"FATAL: Model file not found: {best_model_path}")
        sys.exit(1)

    label_scaler: Optional[MinMaxScaler] = None
    if config.normalize:
        if os.path.exists(scaler_path):
            try:
                label_scaler = joblib.load(scaler_path)
                logging.info(f"Loaded label scaler from {scaler_path}")
            except Exception as e:
                logging.warning(f"Could not load scaler. Error: {e}")
        else:
            logging.warning(f"Warning: {scaler_path} not found. Metrics may be incorrect.")

    logging.info("Loading and processing test data...")
    _, _, test_df, _ = prepare_data(config)

    logging.info(f"Loading tokenizer: {config.model_name}")
    try:
        tokenizer: PreTrainedTokenizer = EsmTokenizer.from_pretrained(config.model_name)
    except Exception as e:
        logging.fatal(f"Could not load tokenizer. Error: {e}")
        sys.exit(1)
        
    test_dataset = MHCSequenceDataset(test_df, tokenizer, config.mhc_col, config.pep_col, config.label_col, config.max_length)
    collate_wrapper = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_wrapper)

    model: nn.Module = ESM2FineTuneModel(esm_model_name=config.model_name).to(device)
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f"Loaded model from {best_model_path}")
    except Exception as e:
        logging.fatal(f"Failed to load model state_dict. Error: {e}")
        sys.exit(1)
        
    criterion: nn.Module = nn.MSELoss()

    logging.info("Running final evaluation on test set...")
    test_loss, test_rmse, test_r2, test_pcc, test_scc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler=label_scaler
    )
    
    logging.info(f"\nTest Results (on original scale):")
    logging.info(f"  Loss: {test_loss:.4f} (scaled)")
    logging.info(f"  RMSE: {test_rmse:.4f}")
    logging.info(f"  R²:   {test_r2:.4f}")
    logging.info(f"  PCC:  {test_pcc:.4f}")
    logging.info(f"  SCC:  {test_scc:.4f}")
    
    pred_df = pd.DataFrame({'true_label': test_targets, 'pred_label': test_preds})
    pred_path = os.path.join(config.output_dir, "test_predictions_eval_only.csv")
    pred_df.to_csv(pred_path, index=False)
    logging.info(f"\n✓ Saved test predictions to {pred_path}")
    
    logging.info("Evaluation complete. Run finished.")

# ===========================================================================
# 9. MAIN EXECUTION
# ===========================================================================

def load_config(config_path: str) -> AppConfig:
    """Loads and validates the YAML config file using Pydantic."""
    try:
        with open(config_path, 'r') as f:
            config_data: Dict[str, Any] = yaml.safe_load(f)
        
        # 🛡️ --- Validate and parse the config ---
        config = AppConfig(**config_data)
        
        # Create output directory *after* validation
        os.makedirs(config.output_dir, exist_ok=True)
        return config
        
    except FileNotFoundError:
        print(f"FATAL: Config file not found at: {config_path}")
        sys.exit(1)
    except ValidationError as e:
        # 🛡️ --- Clean Pydantic error ---
        print(f"FATAL: Configuration error in '{config_path}':\n")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Error parsing config file: {e}")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(description="ESM-2 Fine-tuning (v6 - Pydantic Validation)")
    
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the configuration YAML file.")
    parser.add_argument('--evaluate_only', action='store_true', 
                        help="Skip training and run evaluation.")
    
    args = parser.parse_args()
    
    # 🛡️ --- Load and Validate Config ---
    config = load_config(args.config)
    
    # 🪵 --- Set up logging ---
    setup_logging(config.output_dir)
    
    # 🧬 --- Set Seed ---
    set_seed(config.seed)
    
    logging.info("=" * 70)
    logging.info("ESM-2 Fine-Tuning Script (v6 - Pydantic Validation)")
    logging.info(f"Loaded and validated configuration from: {args.config}")
    logging.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\nUsing device: {device}")
    
    if config.use_amp and device.type == 'cpu':
        logging.warning("Warning: config.use_amp=True but no CUDA found. Disabling AMP.")
        config.use_amp = False
    
    if args.evaluate_only:
        run_evaluation(config, device)
    else:
        run_training(config, device)


if __name__ == "__main__":
    main()