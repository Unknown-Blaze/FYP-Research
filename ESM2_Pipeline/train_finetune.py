"""
ESM-2 Fine-Tuning Script (v3 - Single File Split & TensorBoard)

CHANGE LOG (from v2):
    - Integrated sklearn.model_selection.train_test_split
    - Removed --train, --valid, --test args
    - Added --data_file, --seed, --test_size, --valid_size args
    - Added set_seed() function for reproducibility.
    - Replaced all Weights & Biases (wandb) logging with TensorBoard.
    - Changed default column names to match IEDB headers.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # ⚡ For A100 AMP
from transformers import EsmTokenizer, EsmModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler   # ⚖️ For Normalization
from sklearn.model_selection import train_test_split  # 🆕 For splitting
from scipy.stats import pearsonr, spearmanr
from torch.utils.tensorboard import SummaryWriter  # 📈 For TensorBoard
import warnings
import os
import sys
import random  #  reproducibility

warnings.filterwarnings('ignore')

# ===========================================================================
# 0. REPRODUCIBILITY
# ===========================================================================

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===========================================================================
# 1. MODEL DEFINITION
# ===========================================================================

class ESM2FineTuneModel(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t33_650M_UR50D", hidden_dim=512, dropout=0.2):
        super().__init__()
        print(f"Loading ESM-2 model: {esm_model_name}...")
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        
        esm_dim = self.esm_model.config.hidden_size
        
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
    
    def encode_sequence(self, sequences):
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(self.esm_model.parameters()).device) for k, v in inputs.items()}
        
        outputs = self.esm_model(**inputs)
        embeddings = outputs.last_hidden_state
        
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        
        return embeddings
    
    def forward(self, mhc_sequences, peptide_sequences):
        mhc_emb = self.encode_sequence(mhc_sequences)
        pep_emb = self.encode_sequence(peptide_sequences)
        combined = torch.cat([mhc_emb, pep_emb], dim=1)
        return self.regression_head(combined)

# ===========================================================================
# 2. DATASET
# ===========================================================================

class MHCSequenceDataset(Dataset):
    def __init__(self, df, mhc_col='mhc_seq', peptide_col='peptide', ic50_col='ic50'):
        try:
            self.mhc_seqs = df[mhc_col].tolist()
            self.pep_seqs = df[peptide_col].tolist()
            # Labels are passed in already processed (pIC50 and/or normalized)
            self.labels = df[ic50_col].values.astype(np.float32)
        except KeyError as e:
            print(f"FATAL ERROR: Column not found in CSV: {e}")
            print(f"Please check your CSV or use --mhc_col, --pep_col, --label_col arguments.")
            exit()
        except Exception as e:
            print(f"FATAL ERROR: Could not process DataFrame. Error: {e}")
            print("Check if your CSV file is loaded correctly and columns are correct.")
            exit()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'mhc_sequence': self.mhc_seqs[idx],
            'peptide_sequence': self.pep_seqs[idx],
            'label': self.labels[idx]
        }

# ===========================================================================
# 3. EVALUATION
# ===========================================================================

def evaluate(model, data_loader, criterion, device, scaler=None): # ⚖️ Added scaler
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            mhc_seqs = batch['mhc_sequence']
            pep_seqs = batch['peptide_sequence']
            labels = batch['label'].to(device)

            predictions = model(mhc_seqs, pep_seqs)
            loss = criterion(predictions.squeeze(), labels)
            total_loss += loss.item()

            all_preds.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    
    all_preds = np.array(all_preds).reshape(-1, 1)
    all_targets = np.array(all_targets).reshape(-1, 1)

    # ⚖️ Inverse transform predictions and targets to original scale for metrics
    if scaler:
        try:
            all_preds_orig = scaler.inverse_transform(all_preds)
            all_targets_orig = scaler.inverse_transform(all_targets)
        except ValueError as e:
            print(f"Warning: could not inverse-transform. Metrics will be on scaled (0-1) values. Error: {e}")
            all_preds_orig = all_preds
            all_targets_orig = all_targets
    else:
        all_preds_orig = all_preds
        all_targets_orig = all_targets

    # Calculate metrics on original (e.g., pIC50) scale
    rmse = np.sqrt(mean_squared_error(all_targets_orig, all_preds_orig))
    r2 = r2_score(all_targets_orig, all_preds_orig)
    pcc, _ = pearsonr(all_targets_orig.flatten(), all_preds_orig.flatten())
    scc, _ = spearmanr(all_targets_orig.flatten(), all_preds_orig.flatten())

    # Return predictions in original scale
    return avg_loss, rmse, r2, pcc, scc, all_preds_orig.flatten(), all_targets_orig.flatten()

# ===========================================================================
# 4. MAIN TRAINING & EVALUATION LOOP
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="ESM-2 Fine-tuning (v3 - Single File Split & TensorBoard)")
    
    # --- Data & Model Paths ---
    parser.add_argument('--data_file', type=str, required=True, help="Path to the SINGLE dataset CSV (e.g., from IEDB)")
    parser.add_argument('--model_name', type=str, default="facebook/esm2_t33_650M_UR50D", help="ESM-2 model name")
    parser.add_argument('--output_dir', type=str, default="./checkpoints", help="Folder to save checkpoints and logs")
    
    # --- Data Column Names (CHANGED DEFAULTS) ---
    parser.add_argument('--mhc_col', type=str, default='mhc_sequence', help="Column name for MHC sequences")
    parser.add_argument('--pep_col', type=str, default='peptide', help="Column name for peptide sequences")
    parser.add_argument('--label_col', type=str, default='binding_affinity_nM', help="Column name for the target label (e.g., IC50)")

    # --- Data Splitting & Reproducibility ---
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--test_size', type=float, default=0.1, help="Fraction of data to use for the test set (e.g., 0.1 for 10%)")
    parser.add_argument('--valid_size', type=float, default=0.1, help="Fraction of data to use for the validation set (e.g., 0.1 for 10%)")

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size (increase for A100)")
    parser.add_argument('--accum_steps', type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument('--clip_norm', type=float, default=1.0, help="Gradient clipping norm")
    
    # --- ⚡ A100 / AMP ---
    parser.add_argument('--use_amp', action='store_true', help="Use Automatic Mixed Precision (for A100/H100)")
    
    # --- ⚖️ Data Preprocessing ---
    parser.add_argument('--convert_to_pic50', action='store_true', help="Convert label col from IC50 (nM) to pIC50")
    parser.add_argument('--normalize', action='store_true', help="Apply 0-1 Min-Max scaling to labels")
    
    args = parser.parse_args()
    
    # 🧬 --- Set Seed ---
    set_seed(args.seed)
    
    # 🗂️ Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    last_checkpoint_path = os.path.join(args.output_dir, "last_checkpoint.pt")
    
    # 📈 --- Initialize TensorBoard ---
    log_dir = os.path.join(args.output_dir, 'runs')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    print("=" * 70)
    print("ESM-2 Fine-Tuning Script (v3 - Single File Split & TensorBoard)")
    print("=" * 70)
    print(f"Args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if args.use_amp and device.type == 'cpu':
        print("Warning: --use_amp requested but no CUDA device found. Disabling AMP.")
        args.use_amp = False
    
    # ⚡ Initialize GradScaler for AMP
    scaler = GradScaler(enabled=args.use_amp)

    # 💾 --- Load and Split Data ---
    print("Loading and splitting data...")
    try:
        all_df = pd.read_csv(args.data_file, on_bad_lines='skip') # Added on_bad_lines
        
        # Clean up: remove rows with missing sequences or labels
        initial_count = len(all_df)
        all_df = all_df.dropna(subset=[args.mhc_col, args.pep_col, args.label_col])
        all_df = all_df[all_df[args.pep_col].str.isalpha()] # Ensure only valid peptides
        all_df = all_df[all_df[args.mhc_col].str.isalpha()] # Ensure only valid MHCs
        clean_count = len(all_df)
        print(f"Loaded {initial_count} data points, {clean_count} remaining after cleaning NAs/non-alpha sequences.")
        
        # Split 1: Create test set
        train_val_df, test_df = train_test_split(
            all_df, 
            test_size=args.test_size, 
            random_state=args.seed
        )
        
        # Split 2: Create train and validation from the remaining data
        # Scale validation size to be a fraction of the *original* dataset
        valid_size_scaled = args.valid_size / (1.0 - args.test_size)
        train_df, valid_df = train_test_split(
            train_val_df, 
            test_size=valid_size_scaled, 
            random_state=args.seed
        )

        print(f"  Training set size:   {len(train_df)}")
        print(f"  Validation set size: {len(valid_df)}")
        print(f"  Test set size:       {len(test_df)}")

    except FileNotFoundError:
        print(f"FATAL ERROR: File not found: {args.data_file}")
        writer.close()
        exit()
    except Exception as e:
        print(f"FATAL ERROR: Could not read or split CSV. Error: {e}")
        writer.close()
        exit()


    # 💊 --- pIC50 Conversion ---
    if args.convert_to_pic50:
        print(f"Converting '{args.label_col}' to pIC50...")
        for df in [train_df, valid_df, test_df]:
            if args.label_col not in df.columns:
                print(f"FATAL ERROR: --label_col '{args.label_col}' not found.")
                writer.close()
                exit()
            # Cap at 50,000 nM (a common practice for non-binders)
            df[args.label_col] = df[args.label_col].apply(lambda x: min(float(x), 50000.0))
            # Convert nM to M (1e-9)
            molar_values = df[args.label_col] * 1e-9
            # Calculate pIC50 = -log10(Molar)
            # Add small epsilon to avoid log(0) if value was 0
            df[args.label_col] = -np.log10(molar_values + 1e-12)
        print("✓ Conversion to pIC50 complete.")

    # ⚖️ --- Normalization ---
    label_scaler = None
    if args.normalize:
        print(f"Applying 0-1 normalization on label column: '{args.label_col}'")
        label_scaler = MinMaxScaler()
        
        try:
            # Fit on training data ONLY
            train_labels = train_df[args.label_col].values.reshape(-1, 1)
            label_scaler.fit(train_labels)
            
            # Transform all datasets
            train_df[args.label_col] = label_scaler.transform(train_labels)
            valid_df[args.label_col] = label_scaler.transform(valid_df[args.label_col].values.reshape(-1, 1))
            test_df[args.label_col] = label_scaler.transform(test_df[args.label_col].values.reshape(-1, 1))
            print("✓ Labels normalized.")
        except KeyError:
            print(f"FATAL ERROR: --label_col '{args.label_col}' not found. Cannot normalize.")
            writer.close()
            exit()

    # Create datasets and dataloaders
    print(f"Using columns: MHC='{args.mhc_col}', Peptide='{args.pep_col}', Label='{args.label_col}'")
    train_dataset = MHCSequenceDataset(train_df, mhc_col=args.mhc_col, peptide_col=args.pep_col, ic50_col=args.label_col)
    valid_dataset = MHCSequenceDataset(valid_df, mhc_col=args.mhc_col, peptide_col=args.pep_col, ic50_col=args.label_col)
    test_dataset = MHCSequenceDataset(test_df, mhc_col=args.mhc_col, peptide_col=args.pep_col, ic50_col=args.label_col)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss, and optimizer
    model = ESM2FineTuneModel(esm_model_name=args.model_name).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    best_valid_rmse = float('inf')
    best_epoch = 0
    start_epoch = 0
    
    # 🗂️ --- Checkpoint Resuming ---
    if os.path.exists(last_checkpoint_path):
        print(f"Resuming training from checkpoint: {last_checkpoint_path}")
        try:
            checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_valid_rmse = checkpoint['best_valid_rmse']
            best_epoch = checkpoint.get('best_epoch', 0) # Load best_epoch
            print(f"Resumed from epoch {start_epoch}. Best RMSE so far: {best_valid_rmse:.4f}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Starting from scratch. Error: {e}")
            start_epoch = 0
            best_valid_rmse = float('inf')
            best_epoch = 0


    effective_batch_size = args.batch_size * args.accum_steps
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} (Real) / {effective_batch_size} (Effective)")
    print(f"  Automatic Mixed Precision: {args.use_amp}")
    print("-" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            mhc_seqs = batch['mhc_sequence']
            pep_seqs = batch['peptide_sequence']
            labels = batch['label'].to(device)

            # ⚡ --- AMP Forward Pass ---
            with autocast(enabled=args.use_amp):
                predictions = model(mhc_seqs, pep_seqs)
                loss = criterion(predictions.squeeze(), labels)
            
            loss = loss / args.accum_steps
            train_loss += loss.item() * args.accum_steps
            
            # ⚡ --- AMP Backward Pass ---
            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                
                # ⚡ --- AMP Optimizer Step ---
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader) # Corrected avg train loss
        
        # Validate
        valid_loss, valid_rmse, valid_r2, valid_pcc, valid_scc, _, _ = evaluate(
            model, valid_loader, criterion, device, scaler=label_scaler
        )
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} (scaled)")
        print(f"  Valid Loss: {valid_loss:.4f} (scaled), Valid RMSE: {valid_rmse:.4f}, Valid R²: {valid_r2:.4f}")
        
        # 📈 --- Log metrics to TensorBoard ---
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Metrics/valid_rmse', valid_rmse, epoch)
        writer.add_scalar('Metrics/valid_r2', valid_r2, epoch)
        writer.add_scalar('Metrics/valid_pcc', valid_pcc, epoch)
        writer.add_scalar('Metrics/valid_scc', valid_scc, epoch)
        writer.add_scalar('Params/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step(valid_loss)
        
        # 🗂️ --- Save Best Model ---
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Saved new best model (RMSE: {valid_rmse:.4f}) to {best_model_path}")
        
        # 🗂️ --- Save Last Checkpoint ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_valid_rmse': float(best_valid_rmse),
            'best_epoch': best_epoch, # Save best_epoch
            'args': vars(args) 
        }
        torch.save(checkpoint, last_checkpoint_path)

    # Test
    print("\n" + "=" * 70)
    print("=== Testing ===")
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path} (from epoch {best_epoch + 1})")
    
    test_loss, test_rmse, test_r2, test_pcc, test_scc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler=label_scaler
    )
    
    print(f"\nTest Results (on original pIC50 scale):")
    print(f"  Loss: {test_loss:.4f} (scaled)")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  PCC:  {test_pcc:.4f}")
    print(f"  SCC:  {test_scc:.4f}")
    
    # 📈 --- Log final test metrics and HParams to TensorBoard ---
    hparams = vars(args).copy()
    # Convert any non-loggable types to string
    for key, val in hparams.items():
        if not isinstance(val, (int, float, str, bool)):
            hparams[key] = str(val)

    metrics = {
        'hparam/test_rmse': test_rmse,
        'hparam/test_r2': test_r2,
        'hparam/test_pcc': test_pcc,
        'hparam/test_scc': test_scc,
        'hparam/best_valid_rmse': best_valid_rmse,
        'hparam/best_epoch': best_epoch
    }
    
    writer.add_hparams(hparams, metrics)
    
    # Save test predictions
    pred_df = pd.DataFrame({'true_pIC50': test_targets, 'pred_pIC50': test_preds})
    pred_path = os.path.join(args.output_dir, "test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n✓ Saved test predictions (in pIC50 scale) to {pred_path}")
    
    writer.close()
    print("\nTraining complete. Run finished.")

if __name__ == "__main__":
    main()