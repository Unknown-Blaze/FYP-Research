import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIGURATION (MUST MATCH YOUR GNN/ESM PIPELINE EXACTLY)
# =========================================================
SEED = 42
TEST_SIZE = 0.1
VALID_SIZE = 0.1
PAD_LEN = 385 # Standard for NetMHC/SOTA models

def pad_sequence(seq, target_len=385, pad_char='X'):
    """Ensures MHC sequence is exactly target_len"""
    seq = str(seq).strip()
    if len(seq) > target_len:
        return seq[:target_len]
    else:
        return seq + (pad_char * (target_len - len(seq)))

def to_pic50(ic50):
    """Converts nM to pIC50 (matching your ESM target)"""
    val = float(ic50)
    # Clamp to avoid log(0)
    val = min(val, 50000.0)
    return -np.log10(val * 1e-9 + 1e-12)

def write_sota_format(df, filename):
    """Writes data in the 6-column tab-separated format SOTA expects"""
    with open(filename, 'w') as f:
        # Dummy Header
        f.write("Dummy\tAllele\tDummy\tPeptide\tDummy\tIC50\n")
        
        for idx, row in df.iterrows():
            allele = row['mhc_allele'].replace(' ', '')
            pep = row['peptide']
            
            # CRITICAL: We write pIC50, not raw IC50.
            # This ensures the model learns the same regression task as your ESM model.
            target = to_pic50(row['binding_affinity_nM'])
            
            # We use the index as the first column (UID) to track rows if needed
            f.write(f"{idx}\t{allele}\t0\t{pep}\t0\t{target}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Path to iedb_dataset.csv")
    parser.add_argument('--out_dir', type=str, default='dataset/SOTA_Strict')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # -------------------------------------------------------
    # 1. EXACT CLEANING (COPIED FROM GNN/ESM PIPELINE)
    # -------------------------------------------------------
    print("Applying strict cleaning filters...")
    initial_count = len(df)
    
    # Drop NAs
    df = df.dropna(subset=['binding_affinity_nM', 'mhc_sequence', 'peptide', 'mhc_allele'])
    
    # Filter Alpha (The crucial step for row alignment)
    df = df[df['peptide'].astype(str).str.isalpha()]
    df = df[df['mhc_sequence'].astype(str).str.isalpha()]
    
    print(f"Dropped {initial_count - len(df)} bad rows. Remaining: {len(df)}")

    # -------------------------------------------------------
    # 2. RECONSTRUCT THE SPLIT (SEED 42)
    # -------------------------------------------------------
    print("Reconstructing strict 80/10/10 split...")
    
    # Split 1: Isolate Test Set (10%)
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED
    )
    
    # Split 2: Isolate Validation from Train
    valid_size_scaled = VALID_SIZE / (1.0 - TEST_SIZE)
    train_df, val_df = train_test_split(
        train_val_df, test_size=valid_size_scaled, random_state=SEED
    )
    
    print(f"Split Sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # -------------------------------------------------------
    # 3. SAVE FILES
    # -------------------------------------------------------
    # For SOTA Training, we usually combine Train+Val (90%) 
    # as the model handles its own internal validation split often, 
    # OR we just feed it the strict Train set. 
    # To be safe and maximize data (standard practice), we feed Train+Val as 'train'.
    sota_train_df = pd.concat([train_df, val_df])
    
    print("Writing processed files...")
    
    # A. Write Data Files
    write_sota_format(sota_train_df, f'{args.out_dir}/train_data.txt')
    write_sota_format(test_df,       f'{args.out_dir}/test_data.txt')
    
    # B. Write MHC Sequences (Unique Only)
    unique_mhc = df[['mhc_allele', 'mhc_sequence']].drop_duplicates()
    with open(f'{args.out_dir}/mhc_seqs.txt', 'w') as f:
        f.write("Allele Sequence\n")
        for _, row in unique_mhc.iterrows():
            name = row['mhc_allele'].replace(' ', '')
            seq = pad_sequence(row['mhc_sequence'], target_len=PAD_LEN)
            f.write(f"{name} {seq}\n")

    print(f"✅ Done! Files saved to {args.out_dir}")
    print("You can now point your config.json to these files.")

if __name__ == "__main__":
    main()