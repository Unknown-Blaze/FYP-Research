"""
ESM-2 + ConvNeXt Integration (Minimal Version)

Uses pre-trained ESM-2 from HuggingFace to replace degenerate encoding.
Simple, clean implementation without over-engineering.

Requirements:
    pip install transformers torch pandas scikit-learn

Usage:
    python esm2_simple.py --train final_train_70pct.csv --valid final_valid_10pct.csv --epochs 50
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmModel
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')


class ESM2ConvNeXtModel(nn.Module):
    """
    Simple ESM-2 + ConvNeXt fusion for MHC-peptide binding prediction
    
    Uses pre-trained ESM-2 from HuggingFace (facebook/esm2_t33_650M_UR50D)
    to encode MHC and peptide sequences, then predicts IC50 values.
    """
    
    def __init__(self, esm_model_name="facebook/esm2_t33_650M_UR50D", hidden_dim=256, dropout=0.1):
        super().__init__()
        
        # Load pre-trained ESM-2 from HuggingFace
        print(f"Loading ESM-2 model: {esm_model_name}...")
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        
        # Freeze ESM-2 weights (only train prediction head)
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        esm_dim = self.esm_model.config.hidden_size  # 1280 for 650M model
        
        # Simple prediction head
        self.predictor = nn.Sequential(
            nn.Linear(esm_dim * 2, hidden_dim),  # MHC + Peptide embeddings
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # IC50 prediction
        )
    
    def encode_sequence(self, sequences):
        """Encode sequences using ESM-2 and return mean-pooled embeddings"""
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(self.esm_model.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            embeddings = outputs.last_hidden_state  # [batch, seq_len, esm_dim]
        
        # Mean pooling (exclude padding tokens)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (embeddings * attention_mask).sum(1) / attention_mask.sum(1)
        
        return embeddings
    
    def forward(self, mhc_sequences, peptide_sequences):
        """
        Forward pass
        
        Args:
            mhc_sequences: List of MHC sequences
            peptide_sequences: List of peptide sequences
        
        Returns:
            predictions: [batch, 1] IC50 predictions
        """
        # Encode sequences
        mhc_emb = self.encode_sequence(mhc_sequences)  # [batch, esm_dim]
        pep_emb = self.encode_sequence(peptide_sequences)  # [batch, esm_dim]
        
        # Concatenate and predict
        combined = torch.cat([mhc_emb, pep_emb], dim=1)  # [batch, esm_dim*2]
        predictions = self.predictor(combined)
        
        return predictions


class MHCPeptideDataset(Dataset):
    """Dataset for MHC-peptide binding data"""
    
    def __init__(self, csv_path):
        # Try to read as CSV first, then TSV
        try:
            self.data = pd.read_csv(csv_path)
        except:
            self.data = pd.read_csv(csv_path, sep='\t')
        
        print(f"Loaded {len(self.data)} samples from {csv_path}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Find MHC sequence column (flexible column names)
        self.mhc_col = None
        for col in ['mhc_seq', 'mhc_sequence', 'MHC_sequence', 'allele_seq']:
            if col in self.data.columns:
                self.mhc_col = col
                break
        
        if not self.mhc_col:
            raise ValueError(f"No MHC sequence column found. Available: {list(self.data.columns)}")
        
        # Find peptide column
        self.peptide_col = None
        for col in ['peptide', 'Peptide', 'pep_seq', 'peptide_sequence']:
            if col in self.data.columns:
                self.peptide_col = col
                break
        
        if not self.peptide_col:
            raise ValueError(f"No peptide column found. Available: {list(self.data.columns)}")
        
        print(f"Using MHC column: '{self.mhc_col}'")
        print(f"Using peptide column: '{self.peptide_col}'")
        
        # Find IC50 column (various possible names)
        ic50_cols = [col for col in self.data.columns if 'ic50' in col.lower() or 'affinity' in col.lower() or 'measurement' in col.lower()]
        if not ic50_cols:
            raise ValueError(f"No IC50/affinity column found. Available: {list(self.data.columns)}")
        
        self.ic50_col = ic50_cols[0]
        print(f"Using '{self.ic50_col}' as target column")
        
        # Convert to log scale if needed
        if self.data[self.ic50_col].max() > 50:  # Likely in nM scale
            print("Converting IC50 from nM to pIC50 (log scale)")
            self.data['pIC50'] = -np.log10(self.data[self.ic50_col] / 1e9)
        else:
            self.data['pIC50'] = self.data[self.ic50_col]
        
        # Remove invalid values
        self.data = self.data.dropna(subset=[self.mhc_col, self.peptide_col, 'pIC50'])
        print(f"After cleaning: {len(self.data)} samples")
        print(f"pIC50 range: {self.data['pIC50'].min():.2f} to {self.data['pIC50'].max():.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'mhc_seq': str(row[self.mhc_col]),
            'peptide': str(row[self.peptide_col]),
            'pIC50': torch.tensor(row['pIC50'], dtype=torch.float32)
        }


def collate_fn(batch):
    """Collate function to handle variable-length sequences"""
    return {
        'mhc_sequences': [item['mhc_seq'] for item in batch],
        'peptide_sequences': [item['peptide'] for item in batch],
        'targets': torch.stack([item['pIC50'] for item in batch])
    }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions, targets = [], []
    
    for batch_idx, batch in enumerate(dataloader):
        mhc_seqs = batch['mhc_sequences']
        pep_seqs = batch['peptide_sequences']
        target = batch['targets'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(mhc_seqs, pep_seqs).squeeze(-1)
        loss = criterion(pred, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(pred.detach().cpu().numpy())
        targets.extend(target.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    pcc, _ = pearsonr(targets, predictions)
    scc, _ = spearmanr(targets, predictions)
    
    return avg_loss, rmse, r2, pcc, scc


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0
    predictions, targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            mhc_seqs = batch['mhc_sequences']
            pep_seqs = batch['peptide_sequences']
            target = batch['targets'].to(device)
            
            pred = model(mhc_seqs, pep_seqs).squeeze(-1)
            loss = criterion(pred, target)
            
            total_loss += loss.item()
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    pcc, _ = pearsonr(targets, predictions)
    scc, _ = spearmanr(targets, predictions)
    
    return avg_loss, rmse, r2, pcc, scc, predictions, targets


def main():
    parser = argparse.ArgumentParser(description='ESM-2 + ConvNeXt for MHC-peptide binding prediction')
    parser.add_argument('--train', type=str, required=True, help='Training CSV file')
    parser.add_argument('--valid', type=str, default=None, help='Validation CSV file')
    parser.add_argument('--test', type=str, default=None, help='Test CSV file')
    parser.add_argument('--esm_model', type=str, default='facebook/esm2_t33_650M_UR50D', 
                        help='ESM-2 model from HuggingFace')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--output', type=str, default='esm2_convnext_model.pt', help='Output model path')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n=== Loading Datasets ===")
    train_dataset = MHCPeptideDataset(args.train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    valid_loader = None
    if args.valid:
        valid_dataset = MHCPeptideDataset(args.valid)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    test_loader = None
    if args.test:
        test_dataset = MHCPeptideDataset(args.test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    print("\n=== Creating Model ===")
    model = ESM2ConvNeXtModel(
        esm_model_name=args.esm_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Training loop
    print("\n=== Training ===")
    best_valid_rmse = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_rmse, train_r2, train_pcc, train_scc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        print(f"Train - Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}, "
              f"PCC: {train_pcc:.4f}, SCC: {train_scc:.4f}")
        
        # Validate
        if valid_loader:
            valid_loss, valid_rmse, valid_r2, valid_pcc, valid_scc, _, _ = evaluate(
                model, valid_loader, criterion, device
            )
            
            print(f"Valid - Loss: {valid_loss:.4f}, RMSE: {valid_rmse:.4f}, R²: {valid_r2:.4f}, "
                  f"PCC: {valid_pcc:.4f}, SCC: {valid_scc:.4f}")
            
            # Save best model
            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                torch.save(model.state_dict(), args.output)
                print(f"✓ Saved best model to {args.output}")
    
    # Test
    if test_loader:
        print("\n=== Testing ===")
        if valid_loader:
            model.load_state_dict(torch.load(args.output))
            print(f"Loaded best model from {args.output}")
        
        test_loss, test_rmse, test_r2, test_pcc, test_scc, test_preds, test_targets = evaluate(
            model, test_loader, criterion, device
        )
        
        print(f"Test - Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, "
              f"PCC: {test_pcc:.4f}, SCC: {test_scc:.4f}")
        
        # Save predictions
        results_df = pd.DataFrame({
            'true_pIC50': test_targets,
            'pred_pIC50': test_preds
        })
        results_df.to_csv('esm2_predictions.csv', index=False)
        print("\n✓ Saved predictions to esm2_predictions.csv")


if __name__ == '__main__':
    main()
