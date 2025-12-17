# COMPREHENSIVE TECHNICAL RECONSTRUCTION: SEQUENCE-BASED MHC-PEPTIDE BINDING PREDICTION
## PART 4 OF 5: TRAINING DYNAMICS, OPTIMIZATION, AND EVALUATION

---

## 12. DETAILED TRAINING PROCEDURE

### 12.1 Training Loop Implementation

The training loop follows standard supervised learning with custom adaptations for dual-sequence processing and mixed-precision training:

```python
def run_training(config, device):
    """
    Complete training pipeline with warm-start cross-attention.
    
    Key features:
        - Gradient accumulation for effective large batch sizes
        - Mixed-precision training (FP16) with gradient scaling
        - Separate learning rates for ESM-2 vs. attention
        - Learning rate scheduling with ReduceLROnPlateau
        - Early stopping based on validation RMSE
        - TensorBoard logging
        - Checkpoint saving (best + last)
    """
    
    # Setup TensorBoard
    log_dir = os.path.join(config.output_dir, 'runs')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Prepare data
    train_df, valid_df, test_df, label_scaler = prepare_data(config)
    
    # Create datasets and dataloaders
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    train_dataset = MHCSequenceDataset(train_df, tokenizer, ...)
    valid_dataset = MHCSequenceDataset(valid_df, tokenizer, ...)
    test_dataset = MHCSequenceDataset(test_df, tokenizer, ...)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                               shuffle=True, collate_fn=collate_wrapper)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size,
                               shuffle=False, collate_fn=collate_wrapper)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_wrapper)
    
    # Initialize model
    model = ESM2FineTuneModel(
        esm_model_name=config.model_name,
        num_ca_layers=config.num_ca_layers,
        num_heads=config.num_attention_heads,
        dropout=config.ca_dropout,
        freeze_esm=config.freeze_esm,
        warm_start_gate=config.warm_start_gate
    ).to(device)
    
    # Loss function
    criterion = nn.MSELoss()  # Mean squared error on normalized pIC50
    
    # Optimizer with separate parameter groups
    esm_params = [p for n, p in model.named_parameters() 
                  if 'esm_model' in n and p.requires_grad]
    attention_params = [p for n, p in model.named_parameters() 
                        if 'esm_model' not in n and p.requires_grad]
    
    param_groups = [
        {'params': esm_params, 'lr': config.lr},
        {'params': attention_params, 'lr': config.lr * config.lr_attention_multiplier}
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=config.scheduler_patience,
        factor=0.1,  # Reduce LR by 10× when plateau detected
        verbose=True
    )
    
    # Mixed-precision training
    scaler = GradScaler(enabled=config.use_amp)
    
    # Training state
    best_valid_rmse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        # Iterate over training batches
        for i, (mhc_batch, pep_batch, labels) in enumerate(train_loader):
            labels = labels.to(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=config.use_amp):
                predictions = model(mhc_batch, pep_batch)
                loss = criterion(predictions.squeeze(), labels)
            
            # Gradient accumulation
            loss = loss / config.accum_steps
            train_loss += loss.item() * config.accum_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights every accum_steps batches
            if (i + 1) % config.accum_steps == 0 or (i + 1) == len(train_loader):
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping (prevent exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
                
                # Optimizer step with scaled gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        valid_loss, valid_rmse, valid_r2, valid_pcc, valid_scc, _, _ = evaluate(
            model, valid_loader, criterion, device, scaler=label_scaler
        )
        
        # Logging
        logging.info(f"\nEpoch {epoch+1}/{config.epochs}")
        logging.info(f"  Train Loss: {avg_train_loss:.4f}")
        logging.info(f"  Valid RMSE: {valid_rmse:.4f}, R²: {valid_r2:.4f}, PCC: {valid_pcc:.4f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Metrics/valid_rmse', valid_rmse, epoch)
        writer.add_scalar('Metrics/valid_r2', valid_r2, epoch)
        writer.add_scalar('Metrics/valid_pcc', valid_pcc, epoch)
        
        # Learning rate scheduling
        scheduler.step(valid_loss)
        
        # Save best model
        if valid_rmse < best_valid_rmse:
            best_valid_rmse = valid_rmse
            best_epoch = epoch
            epochs_no_improve = 0
            
            torch.save(model.state_dict(), 
                      os.path.join(config.output_dir, "best_model.pt"))
            logging.info(f"  ✓ Best model saved (RMSE: {valid_rmse:.4f})")
        else:
            epochs_no_improve += 1
        
        # Save last checkpoint (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_valid_rmse': best_valid_rmse
        }, os.path.join(config.output_dir, "last_checkpoint.pt"))
        
        # Early stopping
        if epochs_no_improve >= config.early_stopping_patience:
            logging.info(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(
        os.path.join(config.output_dir, "best_model.pt"),
        map_location=device
    ))
    
    test_loss, test_rmse, test_r2, test_pcc, test_scc, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device, scaler=label_scaler
    )
    
    # Log final results
    logging.info(f"\n{'='*70}")
    logging.info(f"TEST RESULTS:")
    logging.info(f"  RMSE: {test_rmse:.4f}")
    logging.info(f"  R²: {test_r2:.4f}")
    logging.info(f"  PCC: {test_pcc:.4f}")
    logging.info(f"  SCC: {test_scc:.4f}")
    logging.info(f"{'='*70}")
    
    # Save predictions
    pred_df = pd.DataFrame({'true': test_targets, 'pred': test_preds})
    pred_df.to_csv(os.path.join(config.output_dir, "predictions.csv"), index=False)
    
    writer.close()
```

### 12.2 Mixed-Precision Training (Automatic Mixed Precision - AMP)

All experiments use PyTorch's Automatic Mixed Precision to accelerate training on modern GPUs (V100, A100):

**Benefits**:
1. **2× speed improvement**: FP16 operations are 2-8× faster than FP32 on Tensor Cores
2. **50% memory reduction**: Allows larger batch sizes or models
3. **Maintained accuracy**: Gradient scaling prevents underflow

**Implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler
scaler = GradScaler(enabled=config.use_amp)  # Disable for CPU or debugging

# Training loop
for batch in train_loader:
    # Forward pass in FP16
    with autocast(enabled=config.use_amp):
        predictions = model(mhc_batch, pep_batch)
        loss = criterion(predictions, labels)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Unscale before gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step with scaled gradients
    scaler.step(optimizer)
    scaler.update()  # Update scale factor for next iteration
```

**Precision details**:
- Forward pass (model computation): FP16
- Loss computation: FP16
- Gradients: FP32 (after unscaling)
- Optimizer state: FP32
- Model weights: FP32 (master copy), FP16 (during forward pass)

**Gradient scaling mechanism**:
- Multiply loss by scale factor (e.g., 2^16) before backward pass
- This shifts FP16 gradient values away from zero (prevents underflow)
- Unscale gradients before optimizer step to maintain correct update magnitudes

**[INFERRED]**: All experiments used AMP (`use_amp: true` in configs).
**TODO**: Benchmark training time with/without AMP to quantify speedup on specific hardware

### 12.3 Gradient Accumulation

Gradient accumulation simulates larger batch sizes when GPU memory is limited:

**Configuration**:
```yaml
batch_size: 32           # Physical batch size per GPU
accum_steps: 1           # No accumulation (effective batch = 32)
# accum_steps: 4         # Effective batch size = 32 × 4 = 128
```

**Implementation**:
```python
# Normalize loss by accumulation steps
loss = loss / config.accum_steps

# Accumulate gradients
scaler.scale(loss).backward()

# Update weights every accum_steps batches
if (batch_idx + 1) % config.accum_steps == 0:
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**Effect on training**:
- Effective batch size = physical_batch_size × accum_steps
- Reduces gradient noise (more stable updates)
- Enables training with very large batch sizes (e.g., 128, 256) on memory-limited GPUs

**Experiments**:
- Most experiments: accum_steps = 1 (batch_size = 32 is sufficient)
- Some cross-attention experiments: batch_size = 128 without accumulation
- **[INFERRED]**: Large batches may stabilize cross-attention training

### 12.4 Gradient Clipping

Gradient clipping prevents exploding gradients (common in deep networks with attention):

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Mechanism**: If global gradient norm exceeds max_norm (1.0), scale all gradients proportionally:
$$
\mathbf{g}_{\text{clipped}} = \min\left(1, \frac{\text{max\_norm}}{\|\mathbf{g}\|}\right) \cdot \mathbf{g}
$$

**Why necessary**:
- Cross-attention can have unstable gradients early in training
- Very large or very small attention weights → exponential gradient magnitudes
- ESM-2 fine-tuning can cause catastrophic forgetting if gradients too large

**Configuration**: All experiments use `clip_norm: 1.0`

### 12.5 Learning Rate Scheduling

**Scheduler**: ReduceLROnPlateau - reduces learning rate when validation loss plateaus

**Configuration**:
```yaml
scheduler_patience: 3    # Wait 3 epochs before reducing LR
factor: 0.1              # Reduce LR by 10× (multiply by 0.1)
```

**Implementation**:
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',              # Minimize validation loss
    patience=3,
    factor=0.1,
    verbose=True             # Print LR changes
)

# After each epoch
scheduler.step(valid_loss)
```

**Typical training dynamics** (warm-start ESM-2):
```
Epoch 1-10:   LR = 1e-5 (ESM), 1e-4 (attention)
Epoch 11-15:  Validation loss plateaus
Epoch 16:     LR reduced to 1e-6 (ESM), 1e-5 (attention)
Epoch 17-25:  Continued improvement with lower LR
Epoch 26-28:  Validation loss plateaus again
Epoch 29:     LR reduced to 1e-7 (ESM), 1e-6 (attention)
Epoch 30-35:  Minimal improvement, early stopping triggered
```

**[INFERRED]**: Scheduler typically triggers 1-2 LR reductions before early stopping.
**TODO**: Extract exact LR reduction epochs from full training logs

### 12.6 Early Stopping

**Configuration**:
```yaml
early_stopping_patience: 10   # Stop if no improvement for 10 epochs
```

**Implementation**:
```python
if valid_rmse < best_valid_rmse:
    best_valid_rmse = valid_rmse
    epochs_no_improve = 0
    # Save model
else:
    epochs_no_improve += 1

if epochs_no_improve >= config.early_stopping_patience:
    logging.info("Early stopping triggered")
    break
```

**Metric**: Validation RMSE (lower is better)

**Rationale**:
- Prevents overfitting to training set
- Saves computation time (typical stop: epoch 30-40 out of max 100)
- Best model (by validation RMSE) is saved and used for final test evaluation

**Observed behavior**:
- Concatenation models: Stop around epoch 35-42
- Warm-start models: Stop around epoch 30-35 (faster convergence)
- Poor models (ESM-C): Stop around epoch 15-20 (never improve)

---

## 13. EVALUATION METRICS AND INFERENCE

### 13.1 Evaluation Function

```python
def evaluate(model, data_loader, criterion, device, scaler=None):
    """
    Comprehensive model evaluation with multiple regression metrics.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation set
        criterion: Loss function (MSELoss)
        device: CUDA device
        scaler: MinMaxScaler for denormalization (optional)
    
    Returns:
        Tuple of (loss, RMSE, R², PCC, SCC, predictions, targets)
        All metrics on original pIC50 scale (after denormalization)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # Disable gradient computation
        for mhc_batch, pep_batch, labels in data_loader:
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(mhc_batch, pep_batch)
            
            # Loss (on normalized scale)
            loss = criterion(predictions.squeeze(), labels)
            total_loss += loss.item()
            
            # Collect predictions and targets
            all_preds.extend(predictions.squeeze().cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    
    # Convert to numpy arrays
    all_preds_np = np.array(all_preds).reshape(-1, 1)
    all_targets_np = np.array(all_targets).reshape(-1, 1)
    
    # Denormalize to original pIC50 scale
    if scaler:
        all_preds_orig = scaler.inverse_transform(all_preds_np)
        all_targets_orig = scaler.inverse_transform(all_targets_np)
    else:
        all_preds_orig = all_preds_np
        all_targets_orig = all_targets_np
    
    # Compute metrics on original scale
    rmse = np.sqrt(mean_squared_error(all_targets_orig, all_preds_orig))
    r2 = r2_score(all_targets_orig, all_preds_orig)
    pcc, _ = pearsonr(all_targets_orig.flatten(), all_preds_orig.flatten())
    scc, _ = spearmanr(all_targets_orig.flatten(), all_preds_orig.flatten())
    
    return (avg_loss, float(rmse), float(r2), float(pcc), float(scc),
            all_preds_orig.flatten(), all_targets_orig.flatten())
```

### 13.2 Regression Metrics Definitions

#### 13.2.1 Root Mean Squared Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

**Interpretation**:
- Average prediction error in pIC50 units
- Lower is better (0 = perfect prediction)
- RMSE = 0.76 means predictions typically deviate by ±0.76 pIC50 units
- In binding affinity terms: 0.76 pIC50 ≈ 5.75-fold error (10^0.76 ≈ 5.75)

**Benchmark**:
- RMSE < 0.5: Excellent (rare for MHC binding)
- RMSE < 0.8: Strong performance (our best models)
- RMSE < 1.0: Acceptable for practical use
- RMSE > 1.5: Poor (random baseline)

#### 13.2.2 Coefficient of Determination (R²)

$$
R^2 = 1 - \frac{\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}(y_i - \bar{y})^2}
$$

**Interpretation**:
- Proportion of variance explained by model
- Range: (-∞, 1], typically [0, 1] for reasonable models
- R² = 0.72 means model explains 72% of binding affinity variance
- R² = 0 means model predicts constant (mean) value
- R² < 0 means model worse than predicting mean

**Benchmark**:
- R² > 0.8: Excellent
- R² > 0.7: Strong (our best warm-start models)
- R² > 0.6: Good (concatenation baseline)
- R² > 0.5: Acceptable
- R² < 0.5: Weak

#### 13.2.3 Pearson Correlation Coefficient (PCC)

$$
\text{PCC} = \frac{\sum_{i=1}^{N}(y_i - \bar{y})(\hat{y}_i - \bar{\hat{y}})}{\sqrt{\sum_{i=1}^{N}(y_i - \bar{y})^2} \sqrt{\sum_{i=1}^{N}(\hat{y}_i - \bar{\hat{y}})^2}}
$$

**Interpretation**:
- Linear correlation between predictions and true values
- Range: [-1, 1], where 1 = perfect positive correlation
- Measures if predictions track trends (not absolute accuracy)
- PCC = 0.85 is strong linear relationship

**Difference from R²**:
- PCC insensitive to constant bias or scaling errors
- R² penalizes systematic over/under-prediction
- PCC > √R² typically (less stringent)

#### 13.2.4 Spearman Correlation Coefficient (SCC)

$$
\text{SCC} = \text{PCC}(\text{rank}(y), \text{rank}(\hat{y}))
$$

**Interpretation**:
- Rank correlation (monotonic relationship)
- Robust to outliers and non-linear transformations
- For ranking tasks (e.g., "which peptides bind strongest?"), SCC most relevant
- SCC ≈ PCC suggests linear relationship; SCC >> PCC suggests non-linear

**Use case**: If model correctly ranks peptides by binding strength (even if absolute pIC50 wrong), SCC captures this.

### 13.3 Test Set Results Summary (All Experiments)

| Experiment | Model | Architecture | Test RMSE | Test R² | Test PCC | Test SCC |
|-----------|-------|--------------|-----------|---------|----------|----------|
| checkpoints_1 | ESM-2 3B | Concat | 0.8046 | 0.6825 | 0.8273 | 0.8229 |
| checkpoints_2 | ESM-2 650M | Concat | 0.8101 | 0.6781 | 0.8258 | 0.8209 |
| checkpoints_rerun | ESM-2 650M | Concat (rerun) | 0.8100 | 0.6782 | 0.8251 | 0.8210 |
| checkpoints_3 | ESM-C | Concat | 0.9086 | 0.5951 | 0.7817 | 0.7749 |
| checkpoints_5 | ProtT5-XL | Concat | [MISSING] | ~0.66 | [MISSING] | [MISSING] |
| checkpoints_6 | ESM-2 650M | Cross-attn | [MISSING] | [MISSING] | [MISSING] | [MISSING] |
| **checkpoints_esm2_warmstart** | **ESM-2 650M** | **Warm-start (frozen)** | **0.7615** | **0.7156** | **0.8470** | [MISSING] |
| checkpoints_esmc_warmstart | ESM-C | Warm-start | 0.7701 | 0.7091 | 0.8428 | [MISSING] |

**Key findings**:
1. **Best model**: ESM-2 650M warm-start (frozen) - RMSE 0.7615, R² 0.7156
2. **Warm-start improvement**: +5.5% R² over concatenation baseline
3. **Reproducibility**: Concatenation results exactly replicated (R² 0.6781 vs. 0.6782)
4. **Model size**: ESM-2 650M sufficient (3B parameters provide no benefit for concatenation)
5. **Protein LM comparison**: ESM-2 > ProtT5 > ESM-C for MHC binding
6. **Cross-attention benefit**: Warm-start essential for cross-attention to outperform concatenation

### 13.4 Inference Pipeline (Deployment)

For deploying trained models to predict novel MHC-peptide pairs:

```python
def predict_binding_affinity(
    mhc_sequence: str,
    peptide_sequence: str,
    model_checkpoint: str,
    scaler_path: str,
    config: dict
) -> float:
    """
    Predict binding affinity for a single MHC-peptide pair.
    
    Args:
        mhc_sequence: MHC allele amino acid sequence
        peptide_sequence: Peptide amino acid sequence
        model_checkpoint: Path to trained model weights
        scaler_path: Path to label scaler (for denormalization)
        config: Model configuration dict
    
    Returns:
        Predicted pIC50 binding affinity
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ESM2FineTuneModel(**config).to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Tokenize sequences
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    
    mhc_tokens = tokenizer(
        mhc_sequence, return_tensors='pt', truncation=True, max_length=512
    )
    pep_tokens = tokenizer(
        peptide_sequence, return_tensors='pt', truncation=True, max_length=512
    )
    
    # Move to device
    mhc_batch = {k: v.to(device) for k, v in mhc_tokens.items()}
    pep_batch = {k: v.to(device) for k, v in pep_tokens.items()}
    
    # Predict
    with torch.no_grad():
        pred_normalized = model(mhc_batch, pep_batch).cpu().numpy()
    
    # Denormalize
    pred_pic50 = scaler.inverse_transform(pred_normalized.reshape(-1, 1))[0, 0]
    
    return float(pred_pic50)

# Example usage
mhc = "SHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQKM..."  # HLA-A*01:01
peptide = "YLQPRTFLL"  # Example 9-mer

pred_pic50 = predict_binding_affinity(
    mhc, peptide,
    model_checkpoint="./checkpoints_esm2_warmstart/best_model.pt",
    scaler_path="./checkpoints_esm2_warmstart/label_scaler.joblib",
    config={'esm_model_name': 'facebook/esm2_t33_650M_UR50D', ...}
)

print(f"Predicted pIC50: {pred_pic50:.2f}")
print(f"Predicted IC50: {10**(-pred_pic50) * 1e9:.1f} nM")
```

**Batch inference** for high-throughput screening:
```python
def predict_batch(mhc_sequences, peptide_sequences, model, scaler, batch_size=128):
    """Predict binding for multiple MHC-peptide pairs efficiently."""
    dataset = MHCSequenceDataset(
        pd.DataFrame({'mhc_sequence': mhc_sequences, 'peptide': peptide_sequences}),
        tokenizer, ...
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for mhc_batch, pep_batch, _ in loader:
            preds = model(mhc_batch, pep_batch)
            predictions.extend(preds.cpu().numpy())
    
    # Denormalize
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()
```

---

*End of Part 4. Continue to Part 5 for computational requirements, limitations, and future directions.*
