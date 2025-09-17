import pandas as pd
from Bio.Align import PairwiseAligner
from Bio import Align
import re
from sklearn.model_selection import train_test_split # We'll use scikit-learn for a reliable split

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

def main():
    # --- Configuration ---
    # 1. Input File Paths
    combo_train_path = 'datasets_alphafold_finetune/pmhc_finetune/combo_1and2_train.tsv'
    combo_valid_path = 'datasets_alphafold_finetune/pmhc_finetune/combo_1and2_valid.tsv'
    iedb_file_path = 'datasets_alphafold_finetune/pmhc_finetune/mhc_peptide_affinity_structures_iedb.csv'
    
    # 2. Output File Paths
    output_train_path = 'datasets_alphafold_finetune/pmhc_finetune/final_train_set_pIC50.csv'
    output_valid_path = 'datasets_alphafold_finetune/pmhc_finetune/final_valid_set_pIC50.csv'

    # 3. Column Names from your IEDB file
    iedb_mhc_col = 'mhc_sequence'
    iedb_peptide_col = 'peptide'
    
    # --- End of Configuration ---

    print("Loading datasets...")
    try:
        combo_train_df = pd.read_csv(combo_train_path, sep='\t')
        combo_valid_df = pd.read_csv(combo_valid_path, sep='\t')
        iedb_df = pd.read_csv(iedb_file_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a file. Please check your paths. Details: {e}")
        return

    # Combine the two combo dataframes into one large source of structural data
    combo_df = pd.concat([combo_train_df, combo_valid_df], ignore_index=True)
    print(f"Combined combo datasets into a single dataframe with {len(combo_df)} rows.")

    print("Parsing combo dataset...")
    combo_df[['mhc_seq', 'peptide']] = combo_df['target_chainseq'].str.split('/', expand=True, n=1)

    print("Finding matches using biological alignment... (This may take several minutes)")
    merged_rows = []
    
    for i, iedb_row in iedb_df.iterrows():
        peptide_to_match = iedb_row[iedb_peptide_col]
        mhc_to_match = iedb_row[iedb_mhc_col]

        perfect_peptide_matches = combo_df[combo_df['peptide'] == peptide_to_match]

        if not perfect_peptide_matches.empty:
            for j, combo_row in perfect_peptide_matches.iterrows():
                mhc_combo = combo_row['mhc_seq']
                similarity = calculate_similarity(mhc_combo, mhc_to_match)
                
                if similarity >= 95.0:
                    merged_row = combo_row.to_dict()
                    for col_name, value in iedb_row.items():
                        merged_row[f'iedb_{col_name}'] = value
                    merged_row['mhc_similarity_percent'] = similarity
                    merged_rows.append(merged_row)
                    break 
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(iedb_df)} IEDB rows...")

    if not merged_rows:
        print("\nWarning: No matches found! Please double-check your column names and data.")
        return

    print(f"\nSuccessfully found {len(merged_rows)} matching pairs.")
    
    # Convert the list of dictionaries to a final DataFrame
    final_df = pd.DataFrame(merged_rows)
    
    # --- New Step: Split the data into training and validation sets ---
    print("Splitting merged data into 80% training and 20% validation sets...")
    
    train_df, valid_df = train_test_split(
        final_df,
        test_size=0.2,      # 20% of the data will be for validation
        random_state=42     # Using a fixed state ensures the split is the same every time you run it
    )

    print(f"  - New training set size: {len(train_df)} rows")
    print(f"  - New validation set size: {len(valid_df)} rows")

    # Save the final datasets
    train_df.to_csv(output_train_path, index=False)
    valid_df.to_csv(output_valid_path, index=False)
    
    print(f"\nNew training set saved to: {output_train_path}")
    print(f"New validation set saved to: {output_valid_path}")


if __name__ == '__main__':
    main()