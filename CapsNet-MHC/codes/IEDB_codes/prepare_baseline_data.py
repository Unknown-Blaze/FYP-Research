import pandas as pd
import numpy as np
import os
import argparse

def pad_sequence(seq, target_len=385, pad_char='X'):
    """
    Forces sequence to be exactly target_len.
    Truncates if too long, pads with 'X' if too short.
    """
    # Ensure input is string and handle potential lingering non-string types
    seq = str(seq).strip()
    
    if len(seq) > target_len:
        return seq[:target_len]
    else:
        return seq + (pad_char * (target_len - len(seq)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Your SOTA model's CSV")
    parser.add_argument('--out_dir', type=str, default='dataset/My_Baseline_Data')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)

    # --- CRITICAL FIX: Clean the Data ---
    # 1. Drop rows where essential columns are missing (NaN)
    initial_count = len(df)
    df = df.dropna(subset=['mhc_sequence', 'binding_affinity_nM', 'peptide', 'mhc_allele'])
    print(f"Dropped {initial_count - len(df)} rows containing missing values.")

    # 2. Ensure all sequences are treated as strings
    df['mhc_sequence'] = df['mhc_sequence'].astype(str)
    df['peptide'] = df['peptide'].astype(str)
    df['mhc_allele'] = df['mhc_allele'].astype(str)

    # --- 1. Generate MHC Dictionary ---
    print(f"Generating MHC Dictionary (Padding to 385)...")
    
    unique_mhc = df[['mhc_allele', 'mhc_sequence']].drop_duplicates()
    
    with open(f'{args.out_dir}/mhc_seqs.txt', 'w') as f:
        # Dummy header
        f.write("Allele Sequence\n") 
        
        for _, row in unique_mhc.iterrows():
            name = row['mhc_allele'].replace(' ', '') 
            seq = pad_sequence(row['mhc_sequence'], target_len=385) 
            f.write(f"{name} {seq}\n")

    # --- 2. Generate Training Data ---
    print("Generating Training Data (Raw IC50)...")
    
    with open(f'{args.out_dir}/training_data.txt', 'w') as f:
        # Dummy header 
        f.write("Dummy\tAllele\tDummy\tPeptide\tDummy\tIC50\n")
        
        for _, row in df.iterrows():
            allele = row['mhc_allele'].replace(' ', '')
            pep = row['peptide']
            ic50 = row['binding_affinity_nM']
            
            # Format: [0]\t[Allele]\t[2]\t[Peptide]\t[4]\t[IC50]
            f.write(f"0\t{allele}\t0\t{pep}\t0\t{ic50}\n")

    print(f"Done. Files saved to {args.out_dir}")

if __name__ == "__main__":
    main()