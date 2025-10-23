"""
Training Guide for ConvNeXt-MHC and Fusion Models

This script shows you how to:
1. Evaluate pre-trained ConvNeXt on your data
2. Fine-tune ConvNeXt on your pIC50 dataset  
3. Train the fusion model (structure + sequence)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils import input_matrix_generation
from model import convnext_10_10
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: EVALUATE PRE-TRAINED MODEL
# ============================================================================

def evaluate_pretrained_model(csv_file, model_path="./save_model_weight/ConvNeXt-MHC_BA.h5"):
    """
    Evaluate the pre-trained ConvNeXt-MHC model on your dataset
    
    Args:
        csv_file: Your CSV with columns: allele, peptide, 9mer, pIC50
        model_path: Path to pre-trained weights
        
    Returns:
        results: Dict with evaluation metrics
    """
    print("=" * 70)
    print("STEP 1: Evaluating Pre-trained ConvNeXt-MHC")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Check for pIC50 column
    pic50_col = None
    for col in ['pIC50', 'pic50', 'IC50', 'affinity']:
        if col in df.columns:
            pic50_col = col
            break
    
    if pic50_col is None:
        print("❌ No pIC50/affinity column found!")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    print(f"Found labels in column: {pic50_col}")
    print(f"Dataset: {len(df)} samples")
    print(f"pIC50 range: {df[pic50_col].min():.2f} - {df[pic50_col].max():.2f}")
    
    # Generate input matrix
    print("\nGenerating input matrix...")
    X = input_matrix_generation(csv_file)
    y_true = df[pic50_col].values
    
    # Load pre-trained model
    print("\nLoading pre-trained model...")
    model = convnext_10_10(2)
    model.build((1, 20, 9, 21))
    model.load_weights(model_path)
    print("✓ Model loaded")
    
    # Predict
    print("\nRunning predictions...")
    predictions = model.predict(X)
    y_pred = predictions[:, 1]  # BA score (positive class prob)
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    # Regression metrics
    mse = mean_squared_error(y_true_valid, y_pred_valid)
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    
    # Correlation metrics
    pearson_r, pearson_p = pearsonr(y_true_valid, y_pred_valid)
    spearman_r, spearman_p = spearmanr(y_true_valid, y_pred_valid)
    
    # Prediction statistics
    pred_std = y_pred_valid.std()
    pred_range = y_pred_valid.max() - y_pred_valid.min()
    
    results = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pred_std': pred_std,
        'pred_range': pred_range,
        'y_true': y_true_valid,
        'y_pred': y_pred_valid
    }
    
    print(f"\nRegression Metrics:")
    print(f"  MSE:          {mse:.4f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  RMSE:         {rmse:.4f}")
    
    print(f"\nCorrelation Metrics:")
    print(f"  Pearson r:    {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r:   {spearman_r:.4f} (p={spearman_p:.2e})")
    
    print(f"\nPrediction Statistics:")
    print(f"  Std dev:      {pred_std:.4f}")
    print(f"  Range:        {pred_range:.4f}")
    print(f"  Min/Max:      {y_pred_valid.min():.4f} / {y_pred_valid.max():.4f}")
    
    # Compare with your baseline
    print(f"\n" + "=" * 70)
    print("Comparison with Your Baseline")
    print("=" * 70)
    print(f"Your AlphaFold (PAE only):  std=0.02  (collapsed)")
    print(f"ConvNeXt-MHC (pre-trained): std={pred_std:.4f}")
    
    if pred_std > 0.05:
        print("✓ Good! ConvNeXt shows meaningful prediction spread")
    else:
        print("⚠ Warning: Predictions may be collapsed")
    
    return results


# ============================================================================
# STEP 2: FINE-TUNE ON YOUR DATA
# ============================================================================

def finetune_convnext(
    csv_file,
    pretrained_path="./save_model_weight/ConvNeXt-MHC_BA.h5",
    output_path="./finetuned_model.h5",
    epochs=10,
    batch_size=32,
    learning_rate=1e-4,
    validation_split=0.2
):
    """
    Fine-tune pre-trained ConvNeXt-MHC on your pIC50 dataset
    
    Args:
        csv_file: Your CSV with pIC50 labels
        pretrained_path: Path to pre-trained weights
        output_path: Where to save fine-tuned model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Fraction for validation
        
    Returns:
        history: Training history
        model: Fine-tuned model
    """
    print("=" * 70)
    print("STEP 2: Fine-tuning ConvNeXt-MHC on Your Data")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Find pIC50 column
    pic50_col = None
    for col in ['pIC50', 'pic50', 'IC50', 'affinity']:
        if col in df.columns:
            pic50_col = col
            break
    
    if pic50_col is None:
        print("❌ No pIC50 column found!")
        return None, None
    
    # Generate input matrix
    print("\nGenerating input matrix...")
    X = input_matrix_generation(csv_file)
    y = df[pic50_col].values
    
    print(f"Dataset: {len(X)} samples")
    print(f"Input shape: {X.shape}")
    print(f"Label range: {y.min():.2f} - {y.max():.2f}")
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    print(f"\nTrain: {len(X_train)} samples")
    print(f"Val:   {len(X_val)} samples")
    
    # Load pre-trained model
    print("\nLoading pre-trained model...")
    model = convnext_10_10(2)
    model.build((1, 20, 9, 21))
    model.load_weights(pretrained_path)
    print("✓ Pre-trained weights loaded")
    
    # Modify for regression (replace final layer)
    print("\nModifying model for regression...")
    
    # Remove last layer and add regression head
    x = model.layers[-2].output  # Get features before final layer
    regression_head = keras.layers.Dense(1, activation='linear', name='regression_output')(x)
    
    regression_model = keras.Model(inputs=model.input, outputs=regression_head)
    
    # Compile
    print(f"\nCompiling model with lr={learning_rate}...")
    regression_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n" + "=" * 70)
    print(f"Training for {epochs} epochs...")
    print("=" * 70)
    
    history = regression_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    y_pred_val = regression_model.predict(X_val).flatten()
    
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_pearson, _ = pearsonr(y_val, y_pred_val)
    val_spearman, _ = spearmanr(y_val, y_pred_val)
    
    print(f"\nValidation Metrics:")
    print(f"  MSE:       {val_mse:.4f}")
    print(f"  MAE:       {val_mae:.4f}")
    print(f"  Pearson:   {val_pearson:.4f}")
    print(f"  Spearman:  {val_spearman:.4f}")
    print(f"  Pred std:  {y_pred_val.std():.4f}")
    
    print(f"\n✓ Fine-tuned model saved to: {output_path}")
    
    return history, regression_model


# ============================================================================
# STEP 3: TRAIN FUSION MODEL
# ============================================================================

def train_fusion_model(
    fusion_data_path,
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    output_path="./fusion_model_weights.h5"
):
    """
    Train fusion model combining structure + sequence features
    
    Args:
        fusion_data_path: Path to fusion_dataset.npz (from extract_features.py)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_path: Where to save trained fusion model
        
    Returns:
        history: Training history
    """
    print("=" * 70)
    print("STEP 3: Training Fusion Model (Structure + Sequence)")
    print("=" * 70)
    
    # Load fusion dataset
    print(f"\nLoading fusion data from: {fusion_data_path}")
    data = np.load(fusion_data_path)
    
    print("\nDataset contents:")
    for key in data.files:
        if isinstance(data[key], np.ndarray):
            print(f"  {key:<25} shape: {data[key].shape}")
    
    # Extract features and labels
    convnext_features = data['convnext_features']
    
    # Check if AlphaFold features available
    if 'alphafold_features' in data.files:
        alphafold_features = data['alphafold_features']
        print(f"\n✓ Using both structure and sequence features")
    else:
        print(f"\n⚠ No AlphaFold features found, using ConvNeXt only")
        alphafold_features = None
    
    # Labels
    if 'pic50_labels' in data.files:
        y = data['pic50_labels']
    else:
        print("❌ No pIC50 labels found!")
        return None
    
    # Split train/val
    if alphafold_features is not None:
        X_conv_train, X_conv_val, X_af_train, X_af_val, y_train, y_val = train_test_split(
            convnext_features, alphafold_features, y,
            test_size=0.2, random_state=42
        )
    else:
        X_conv_train, X_conv_val, y_train, y_val = train_test_split(
            convnext_features, y, test_size=0.2, random_state=42
        )
        X_af_train = X_af_val = None
    
    print(f"\nTrain samples: {len(y_train)}")
    print(f"Val samples:   {len(y_val)}")
    
    # Build fusion model
    print("\nBuilding fusion model...")
    
    # ConvNeXt input
    conv_input = keras.Input(shape=(convnext_features.shape[1],), name='convnext_input')
    
    if alphafold_features is not None:
        # AlphaFold input
        af_input = keras.Input(shape=(alphafold_features.shape[1],), name='alphafold_input')
        
        # Process structure features
        af_processed = keras.layers.Dense(128, activation='relu')(af_input)
        af_processed = keras.layers.Dropout(0.2)(af_processed)
        af_processed = keras.layers.Dense(64, activation='relu')(af_processed)
        
        # Binary classifier from structure
        binary_logits = keras.layers.Dense(2, activation='softmax', name='binary_class')(af_processed)
        binding_prob = keras.layers.Lambda(lambda x: x[:, 1:2], name='binding_prob')(binary_logits)
        
        # Process sequence features
        conv_processed = keras.layers.Dense(64, activation='relu')(conv_input)
        conv_processed = keras.layers.Dropout(0.2)(conv_processed)
        
        # Raw BA prediction
        raw_ba = keras.layers.Dense(1, activation='sigmoid', name='raw_ba')(conv_processed)
        
        # Fusion: multiply by binding probability (gating)
        fused = keras.layers.Multiply(name='fusion')([raw_ba, binding_prob])
        
        # Final prediction
        final_output = keras.layers.Dense(1, activation='linear', name='final_output')(fused)
        
        model = keras.Model(
            inputs=[conv_input, af_input],
            outputs=final_output
        )
        
        X_train = [X_conv_train, X_af_train]
        X_val = [X_conv_val, X_af_val]
        
    else:
        # Sequence-only model
        x = keras.layers.Dense(128, activation='relu')(conv_input)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(1, activation='linear')(x)
        
        model = keras.Model(inputs=conv_input, outputs=output)
        
        X_train = X_conv_train
        X_val = X_conv_val
    
    # Compile
    print(f"\nCompiling model with lr={learning_rate}...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    # Train
    print(f"\n" + "=" * 70)
    print(f"Training fusion model for {epochs} epochs...")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Final Fusion Model Evaluation")
    print("=" * 70)
    
    y_pred_val = model.predict(X_val).flatten()
    
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_pearson, _ = pearsonr(y_val, y_pred_val)
    val_spearman, _ = spearmanr(y_val, y_pred_val)
    pred_std = y_pred_val.std()
    
    print(f"\nValidation Metrics:")
    print(f"  MSE:       {val_mse:.4f}")
    print(f"  MAE:       {val_mae:.4f}")
    print(f"  Pearson:   {val_pearson:.4f}")
    print(f"  Spearman:  {val_spearman:.4f}")
    print(f"  Pred std:  {pred_std:.4f}")
    
    print(f"\n✓ Fusion model saved to: {output_path}")
    
    return history


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("ConvNeXt-MHC Training & Evaluation Workflow")
    print("=" * 70)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python train_evaluate.py STEP [arguments]")
        print("\nSteps:")
        print("  1. evaluate <csv_file>")
        print("     Evaluate pre-trained model on your data")
        print("\n  2. finetune <csv_file> [epochs]")
        print("     Fine-tune on your pIC50 dataset")
        print("\n  3. fusion <fusion_data.npz> [epochs]")
        print("     Train fusion model")
        print("\nExamples:")
        print("  python train_evaluate.py evaluate ./data_set/example.csv")
        print("  python train_evaluate.py finetune my_data.csv 20")
        print("  python train_evaluate.py fusion fusion_dataset.npz 30")
        sys.exit(0)
    
    step = sys.argv[1].lower()
    
    if step == "evaluate" or step == "1":
        if len(sys.argv) < 3:
            print("❌ Please provide CSV file")
            print("Usage: python train_evaluate.py evaluate <csv_file>")
            sys.exit(1)
        
        csv_file = sys.argv[2]
        results = evaluate_pretrained_model(csv_file)
        
    elif step == "finetune" or step == "2":
        if len(sys.argv) < 3:
            print("❌ Please provide CSV file")
            print("Usage: python train_evaluate.py finetune <csv_file> [epochs]")
            sys.exit(1)
        
        csv_file = sys.argv[2]
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        
        history, model = finetune_convnext(csv_file, epochs=epochs)
        
    elif step == "fusion" or step == "3":
        if len(sys.argv) < 3:
            print("❌ Please provide fusion dataset file")
            print("Usage: python train_evaluate.py fusion <fusion_data.npz> [epochs]")
            sys.exit(1)
        
        fusion_file = sys.argv[2]
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        
        history = train_fusion_model(fusion_file, epochs=epochs)
        
    else:
        print(f"❌ Unknown step: {step}")
        print("Valid steps: evaluate, finetune, fusion")
        sys.exit(1)
