import pandas as pd
from Bio.Align import PairwiseAligner
from Bio import Align
import re
import numpy as np
from sklearn.model_selection import train_test_split

def clean_sequence(sequence):
    """
    Cleans a protein sequence by removing non-amino acid characters.
    """
    cleaned_seq = str(sequence).strip().upper()
    cleaned_seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', cleaned_seq)
    return cleaned_seq

def calculate_similarity(seq1, seq2):
    """
    Calculates similarity based on a standard Needleman-Wunsch global alignment.
    """
    seq1_clean = clean_sequence(seq1)
    seq2_clean = clean_sequence(seq2)

    if not seq1_clean or not seq2_clean:
        return 0.0

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    try:
        alignment_score = aligner.score(seq1_clean, seq2_clean)
        score_seq1_self = aligner.score(seq1_clean, seq1_clean)
        score_seq2_self = aligner.score(seq2_clean, seq2_clean)
        
        if min(score_seq1_self, score_seq2_self) == 0:
            return 0.0

        normalized_score = (alignment_score / min(score_seq1_self, score_seq2_self)) * 100
        return max(0, normalized_score)
    except ValueError as e:
        print(f"  [Warning] Could not align sequences: {e}. Skipping.")
        return 0.0

def create_test_dataset(combo_test_path, iedb_file_path, output_test_path, 
                       similarity_threshold=95.0, iedb_mhc_col='mhc_sequence', 
                       iedb_peptide_col='peptide'):
    """
    Creates a test dataset by merging combo dataset with IEDB affinity data.
    This is specifically designed for test sets that need IEDB affinity information.
    """
    print("Loading test dataset...")
    try:
        combo_test_df = pd.read_csv(combo_test_path, sep='\t')
        iedb_df = pd.read_csv(iedb_file_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. Please check your paths. Details: {e}")
        return None

    print(f"Loaded combo test dataset with {len(combo_test_df)} rows.")
    print(f"Loaded IEDB dataset with {len(iedb_df)} rows.")

    print("Parsing combo dataset...")
    combo_test_df[['mhc_seq', 'peptide']] = combo_test_df['target_chainseq'].str.split('/', expand=True, n=1)

    print("Finding matches using biological alignment... (This may take several minutes)")
    merged_rows = []
    
    for i, iedb_row in iedb_df.iterrows():
        peptide_to_match = iedb_row[iedb_peptide_col]
        mhc_to_match = iedb_row[iedb_mhc_col]

        # Find perfect peptide matches
        perfect_peptide_matches = combo_test_df[combo_test_df['peptide'] == peptide_to_match]

        if not perfect_peptide_matches.empty:
            for j, combo_row in perfect_peptide_matches.iterrows():
                mhc_combo = combo_row['mhc_seq']
                similarity = calculate_similarity(mhc_combo, mhc_to_match)
                
                if similarity >= similarity_threshold:
                    merged_row = combo_row.to_dict()
                    # Add all IEDB columns with 'iedb_' prefix
                    for col_name, value in iedb_row.items():
                        merged_row[f'iedb_{col_name}'] = value
                    merged_row['mhc_similarity_percent'] = similarity
                    merged_rows.append(merged_row)
                    break  # Take the first good match per IEDB entry
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(iedb_df)} IEDB rows...")

    if not merged_rows:
        print("\nWarning: No matches found! Please double-check your column names and data.")
        return None

    print(f"\nSuccessfully found {len(merged_rows)} matching pairs for test set.")
    
    # Convert to DataFrame
    final_test_df = pd.DataFrame(merged_rows)
    
    # Save the test dataset
    final_test_df.to_csv(output_test_path, index=False)
    print(f"Test set saved to: {output_test_path}")
    
    return final_test_df

def create_prediction_targets_from_structural(structural_file_path, output_path, 
                                             default_mhc_class='I'):
    """
    Creates prediction targets from a structural dataset (like your 1k5n example).
    This converts structural data to prediction format without requiring IEDB matches.
    """
    print(f"Loading structural dataset from {structural_file_path}...")
    try:
        struct_df = pd.read_csv(structural_file_path, sep='\t')
    except FileNotFoundError as e:
        print(f"Error: Could not find file {structural_file_path}. Details: {e}")
        return None

    print(f"Loaded {len(struct_df)} structural entries.")
    
    # Parse the target_chainseq to get MHC and peptide
    struct_df[['mhc_seq', 'peptide']] = struct_df['target_chainseq'].str.split('/', expand=True, n=1)
    
    prediction_targets = []
    for _, row in struct_df.iterrows():
        target_dict = {
            'targetid': row['targetid'],
            'target_chainseq': row['target_chainseq'],
            'templates_alignfile': row['templates_alignfile'],
            'peptide_positions': row['peptide_positions'],
            'mhc_seq': row['mhc_seq'],
            'peptide': row['peptide'],
            'iedb_mhc_class': default_mhc_class,  # Default to Class I
            'iedb_peptide_len': len(row['peptide']),
        }
        
        # Add other available columns
        for col in ['native_pdbfile', 'native_alignstring', 'binder', 'native_exists', 'binder_class']:
            if col in row:
                target_dict[col] = row[col]
        
        prediction_targets.append(target_dict)
    
    prediction_df = pd.DataFrame(prediction_targets)
    prediction_df.to_csv(output_path, index=False)
    print(f"Prediction targets saved to: {output_path}")
    
    return prediction_df

def create_hcv_test_targets(hcv_file_path, iedb_file_path, output_path,
                           mhc_allele_map=None):
    """
    Creates test targets from HCV polg dataset by matching with IEDB data.
    """
    print(f"Loading HCV dataset from {hcv_file_path}...")
    try:
        hcv_df = pd.read_csv(hcv_file_path, sep='\t')
        iedb_df = pd.read_csv(iedb_file_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. Details: {e}")
        return None

    print(f"Loaded {len(hcv_df)} HCV entries and {len(iedb_df)} IEDB entries.")
    
    # Default MHC allele mapping
    if mhc_allele_map is None:
        mhc_allele_map = {
            'A*02:01': 'I',
            'A*01:01': 'I', 
            'A*03:01': 'I',
            'B*07:02': 'I',
            'B*08:01': 'I',
        }
    
    # Parse HCV data
    hcv_df[['mhc_seq', 'peptide']] = hcv_df['target_chainseq'].str.split('/', expand=True, n=1)
    
    prediction_targets = []
    matches_found = 0
    
    for _, hcv_row in hcv_df.iterrows():
        peptide = hcv_row['peptide']
        mhc_allele = hcv_row['mhc']
        
        # Look for this peptide in IEDB data
        iedb_matches = iedb_df[iedb_df['peptide'] == peptide]
        
        target_dict = {
            'targetid': hcv_row['targetid'],
            'target_chainseq': hcv_row['target_chainseq'],
            'templates_alignfile': hcv_row['templates_alignfile'],
            'peptide_positions': f"{len(hcv_row['mhc_seq'])};{len(hcv_row['mhc_seq'])+1};{len(hcv_row['mhc_seq'])+2};{len(hcv_row['mhc_seq'])+3};{len(hcv_row['mhc_seq'])+4};{len(hcv_row['mhc_seq'])+5};{len(hcv_row['mhc_seq'])+6};{len(hcv_row['mhc_seq'])+7};{len(hcv_row['mhc_seq'])+8};{len(hcv_row['mhc_seq'])+9}",  # Assuming 10-mer at the end
            'mhc_seq': hcv_row['mhc_seq'],
            'peptide': peptide,
            'iedb_mhc_class': mhc_allele_map.get(mhc_allele, 'I'),
            'iedb_peptide_len': len(peptide),
        }
        
        # If we found IEDB matches, use the first one
        if not iedb_matches.empty:
            iedb_match = iedb_matches.iloc[0]
            for col in iedb_match.index:
                target_dict[f'iedb_{col}'] = iedb_match[col]
            matches_found += 1
        else:
            # Set default values if no IEDB match
            target_dict['iedb_binding_affinity_nM'] = np.nan
        
        prediction_targets.append(target_dict)
    
    print(f"Found IEDB matches for {matches_found}/{len(hcv_df)} HCV entries.")
    
    prediction_df = pd.DataFrame(prediction_targets)
    prediction_df.to_csv(output_path, index=False)
    print(f"HCV prediction targets saved to: {output_path}")
    
    return prediction_df

def main():
    """
    Main function - runs Option 3 (HCV + IEDB matching) and then splits the data.
    """
    
    # --- Step 1: Create HCV test targets with IEDB matching ---
    print("=== Step 1: Creating HCV test targets with IEDB matching ===")
    hcv_file_path = 'examples/pmhc_hcv_polg_10mers/targets.tsv'  # Your HCV format
    iedb_file_path = 'datasets_alphafold_finetune/pmhc_finetune/mhc_peptide_affinity_structures_iedb.csv'
    hcv_output_path = 'hcv_test_targets_with_iedb.csv'
    
    # Run the HCV matching
    hcv_test_df = create_hcv_test_targets(hcv_file_path, iedb_file_path, hcv_output_path)
    
    # Validate the HCV dataset
    if hcv_test_df is not None:
        print(f"\n=== HCV Test Dataset Summary ===")
        print(f"Total entries: {len(hcv_test_df)}")
        print(f"Columns: {list(hcv_test_df.columns)}")
        
        # Check for IEDB affinity data
        has_affinity = hcv_test_df['iedb_binding_affinity_nM'].notna().sum()
        print(f"Entries with IEDB affinity data: {has_affinity}/{len(hcv_test_df)}")
        
        # Show some example rows
        print(f"\nFirst 3 rows:")
        print(hcv_test_df[['targetid', 'peptide', 'iedb_mhc_class', 'iedb_peptide_len', 'iedb_binding_affinity_nM']].head(3))
        
        # Check affinity distribution if available
        if has_affinity > 0:
            affinities = hcv_test_df['iedb_binding_affinity_nM'].dropna()
            pic50s = 9.0 - np.log10(affinities.clip(lower=0.01))
            print(f"\nAffinity data summary:")
            print(f"  pIC50 range: {pic50s.min():.2f} to {pic50s.max():.2f}")
            print(f"  pIC50 mean: {pic50s.mean():.2f}")
            print(f"  pIC50 std: {pic50s.std():.2f}")
        
        # --- Step 2: Split the HCV data if we have enough entries with affinity data ---
        if has_affinity >= 10:  # Need at least 10 entries to split meaningfully
            print(f"\n=== Step 2: Splitting HCV data into train/val/test ===")
            
            # Filter to only entries with affinity data for splitting
            hcv_with_affinity = hcv_test_df[hcv_test_df['iedb_binding_affinity_nM'].notna()].copy()
            
            # Split: 60% train, 20% validation, 20% test
            train_df, temp_df = train_test_split(hcv_with_affinity, test_size=0.4, random_state=42)
            valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            print(f"Split HCV data with affinity into:")
            print(f"  - Training: {len(train_df)} samples")
            print(f"  - Validation: {len(valid_df)} samples") 
            print(f"  - Test: {len(test_df)} samples")
            
            # Save the HCV splits
            train_df.to_csv('hcv_train_set.csv', index=False)
            valid_df.to_csv('hcv_valid_set.csv', index=False)
            test_df.to_csv('hcv_test_set.csv', index=False)
            
            print("Saved HCV data splits!")
            
            # Show stats for each split
            for split_name, split_df in [("Training", train_df), ("Validation", valid_df), ("Test", test_df)]:
                affinities = split_df['iedb_binding_affinity_nM'].dropna()
                pic50s = 9.0 - np.log10(affinities.clip(lower=0.01))
                print(f"\n{split_name} set pIC50 stats:")
                print(f"  Range: {pic50s.min():.2f} to {pic50s.max():.2f}")
                print(f"  Mean: {pic50s.mean():.2f} ± {pic50s.std():.2f}")
        else:
            print(f"\nNot enough entries with affinity data ({has_affinity}) to split. Need at least 10.")
            print("The full HCV dataset is saved as a single file for prediction.")
        
    else:
        print("Failed to create HCV test dataset!")
    
    # --- Step 3: Also split existing training data if available ---
    print(f"\n=== Step 3: Creating held-out test set from existing training data ===")
    train_file_path = 'datasets_alphafold_finetune/pmhc_finetune/final_train_set_pIC50.csv'
    
    try:
        full_train_df = pd.read_csv(train_file_path)
        
        # Split: 60% train, 20% validation, 20% test
        train_df, temp_df = train_test_split(full_train_df, test_size=0.4, random_state=42)
        valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        print(f"Split existing training data into:")
        print(f"  - Training: {len(train_df)} samples")
        print(f"  - Validation: {len(valid_df)} samples") 
        print(f"  - Test: {len(test_df)} samples")
        
        # Save the new splits with clear names
        train_df.to_csv('datasets_alphafold_finetune/pmhc_finetune/final_train_60pct.csv', index=False)
        valid_df.to_csv('datasets_alphafold_finetune/pmhc_finetune/final_valid_20pct.csv', index=False)
        test_df.to_csv('datasets_alphafold_finetune/pmhc_finetune/final_test_20pct.csv', index=False)
        
        print("Saved existing data splits!")
        
        # Show stats for the main training data splits
        for split_name, split_df in [("Main Training", train_df), ("Main Validation", valid_df), ("Main Test", test_df)]:
            if 'iedb_binding_affinity_nM' in split_df.columns:
                affinities = split_df['iedb_binding_affinity_nM'].dropna()
                if len(affinities) > 0:
                    pic50s = 9.0 - np.log10(affinities.clip(lower=0.01))
                    print(f"\n{split_name} set pIC50 stats:")
                    print(f"  Range: {pic50s.min():.2f} to {pic50s.max():.2f}")
                    print(f"  Mean: {pic50s.mean():.2f} ± {pic50s.std():.2f}")
        
    except FileNotFoundError:
        print("Existing training file not found. Only HCV data processed.")
    
    print(f"\n=== Summary of files created ===")
    print("HCV data (if matching succeeded):")
    print("  - hcv_test_targets_with_iedb.csv (all HCV data with IEDB matches)")
    print("  - hcv_train_set.csv (if enough data for splitting)")
    print("  - hcv_valid_set.csv (if enough data for splitting)")  
    print("  - hcv_test_set.csv (if enough data for splitting)")
    print("\nExisting training data splits (if file found):")
    print("  - final_train_60pct.csv")
    print("  - final_valid_20pct.csv")
    print("  - final_test_20pct.csv")

if __name__ == '__main__':
    main()