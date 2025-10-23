"""
ESM-2 Verification Script

This script:
1. Checks if your dataset has the right structure
2. Tests ESM-2 with a small sample (5 examples)
3. Verifies the entire pipeline works
4. Shows you exactly what's happening

Run this BEFORE full training to ensure everything works!
"""

import sys
import pandas as pd
import numpy as np

print("=" * 70)
print("ESM-2 VERIFICATION SCRIPT")
print("=" * 70)

# Step 1: Check if required packages are installed
print("\n[Step 1/5] Checking dependencies...")
missing_packages = []

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    missing_packages.append("torch")
    print("✗ PyTorch not found")

try:
    from transformers import EsmTokenizer, EsmModel
    print("✓ Transformers (HuggingFace)")
except ImportError:
    missing_packages.append("transformers")
    print("✗ Transformers not found")

try:
    from sklearn.metrics import mean_squared_error, r2_score
    print("✓ scikit-learn")
except ImportError:
    missing_packages.append("scikit-learn")
    print("✗ scikit-learn not found")

try:
    from scipy.stats import pearsonr
    print("✓ scipy")
except ImportError:
    missing_packages.append("scipy")
    print("✗ scipy not found")

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print("\nInstall them with:")
    print("  pip install torch transformers scikit-learn scipy")
    sys.exit(1)

print("\n✅ All dependencies installed!")

# Step 2: Check dataset files
print("\n[Step 2/5] Looking for dataset files...")

# Possible locations for your datasets
dataset_paths = [
    "/home3/rudrapra001/FYP-Research/alphafold_finetune/final_train_70pct.csv",
    "/home3/rudrapra001/FYP-Research/alphafold_finetune/test_small.csv",
    "/home3/rudrapra001/FYP-Research/alphafold_finetune/datasets_alphafold_finetune/pmhc_paper_class_1_test_set_new.tsv",
]

available_datasets = []
for path in dataset_paths:
    try:
        if path.endswith('.tsv'):
            df = pd.read_csv(path, sep='\t', nrows=5)
        else:
            df = pd.read_csv(path, nrows=5)
        available_datasets.append((path, df))
        print(f"✓ Found: {path}")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"✗ Not found: {path}")
    except Exception as e:
        print(f"⚠ Error reading {path}: {e}")

if not available_datasets:
    print("\n❌ No datasets found!")
    print("\nPlease provide a CSV/TSV file with at least these columns:")
    print("  - mhc_seq (or mhc_sequence): MHC sequence")
    print("  - peptide: Peptide sequence")
    print("  - ic50 (or binding_affinity, affinity_nM): IC50 value")
    sys.exit(1)

print(f"\n✅ Found {len(available_datasets)} dataset(s)")

# Step 3: Analyze first available dataset
print("\n[Step 3/5] Analyzing dataset structure...")
test_path, test_df = available_datasets[0]
print(f"\nUsing: {test_path}")
print(f"Shape: {test_df.shape}")
print(f"\nFirst few rows:")
print(test_df.head())

# Check for required columns
print("\n[Step 3.1] Checking for required columns...")

# MHC sequence column
mhc_col = None
for col in ['mhc_seq', 'mhc_sequence', 'MHC_sequence', 'allele_seq']:
    if col in test_df.columns:
        mhc_col = col
        break

if mhc_col:
    print(f"✓ MHC column found: '{mhc_col}'")
    print(f"  Example: {test_df[mhc_col].iloc[0][:50]}...")
else:
    print("❌ No MHC sequence column found!")
    print(f"Available columns: {list(test_df.columns)}")
    sys.exit(1)

# Peptide column
peptide_col = None
for col in ['peptide', 'Peptide', 'pep_seq', 'peptide_sequence']:
    if col in test_df.columns:
        peptide_col = col
        break

if peptide_col:
    print(f"✓ Peptide column found: '{peptide_col}'")
    print(f"  Example: {test_df[peptide_col].iloc[0]}")
else:
    print("❌ No peptide column found!")
    sys.exit(1)

# IC50/Affinity column
ic50_col = None
for col in ['ic50', 'IC50', 'binding_affinity', 'affinity_nM', 'iedb_binding_affinity_nM', 'measurement_value']:
    if col in test_df.columns:
        ic50_col = col
        break

if ic50_col:
    print(f"✓ IC50 column found: '{ic50_col}'")
    print(f"  Values: {test_df[ic50_col].iloc[:3].tolist()}")
else:
    print("⚠ No IC50/affinity column found - will use dummy values for testing")
    test_df['ic50'] = 50.0  # Dummy value
    ic50_col = 'ic50'

print("\n✅ Dataset structure is valid!")

# Step 4: Test ESM-2 model loading
print("\n[Step 4/5] Testing ESM-2 model...")
print("Loading ESM-2 (this may take 1-2 minutes)...")

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name)
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    print(f"✓ ESM-2 model loaded successfully!")
    print(f"  Model: {model_name}")
    print(f"  Embedding dimension: {esm_model.config.hidden_size}")
    
except Exception as e:
    print(f"❌ Failed to load ESM-2: {e}")
    print("\nThis might be due to:")
    print("  1. No internet connection (model downloads from HuggingFace)")
    print("  2. Insufficient memory")
    print("  3. Wrong transformers version")
    sys.exit(1)

# Step 5: Test encoding with real data
print("\n[Step 5/5] Testing ESM-2 encoding with your data...")

# Take first 3 samples
sample_df = test_df.head(3)

for idx, row in sample_df.iterrows():
    mhc_seq = str(row[mhc_col])
    pep_seq = str(row[peptide_col])
    ic50_val = float(row[ic50_col])
    
    print(f"\n--- Sample {idx+1} ---")
    print(f"MHC length: {len(mhc_seq)}")
    print(f"Peptide: {pep_seq} (length: {len(pep_seq)})")
    print(f"IC50: {ic50_val} nM")
    
    try:
        # Encode MHC
        mhc_inputs = tokenizer([mhc_seq], return_tensors="pt", padding=True, truncation=True, max_length=512)
        mhc_inputs = {k: v.to(device) for k, v in mhc_inputs.items()}
        
        with torch.no_grad():
            mhc_outputs = esm_model(**mhc_inputs)
            mhc_emb = mhc_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        
        print(f"✓ MHC encoded: {mhc_emb.shape}")
        
        # Encode peptide
        pep_inputs = tokenizer([pep_seq], return_tensors="pt", padding=True, truncation=True, max_length=512)
        pep_inputs = {k: v.to(device) for k, v in pep_inputs.items()}
        
        with torch.no_grad():
            pep_outputs = esm_model(**pep_inputs)
            pep_emb = pep_outputs.last_hidden_state.mean(dim=1)
        
        print(f"✓ Peptide encoded: {pep_emb.shape}")
        
        # Concatenate
        combined = torch.cat([mhc_emb, pep_emb], dim=1)
        print(f"✓ Combined features: {combined.shape}")
        
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
        sys.exit(1)

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE - Everything works!")
print("=" * 70)

print("\n📊 Summary:")
print(f"  ✓ Dataset: {test_path}")
print(f"  ✓ MHC column: '{mhc_col}'")
print(f"  ✓ Peptide column: '{peptide_col}'")
print(f"  ✓ IC50 column: '{ic50_col}'")
print(f"  ✓ ESM-2 model: {model_name}")
print(f"  ✓ Embedding dimension: {esm_model.config.hidden_size}")
print(f"  ✓ Device: {device}")

print("\n🚀 You're ready to train! Use this command:")
print(f"\n  python esm2_simple.py \\")
print(f"    --train {test_path} \\")
print(f"    --epochs 10 \\")
print(f"    --batch_size 8")

print("\n💡 Tips:")
print("  - Start with small epochs (10) to test")
print("  - Use batch_size=8 if you get memory errors")
print("  - The script will automatically detect column names")
print("  - IC50 values will be converted to pIC50 (log scale)")

print("\n" + "=" * 70)
