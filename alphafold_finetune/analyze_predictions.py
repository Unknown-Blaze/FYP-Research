import os
import numpy as np
import pandas as pd
import glob
import re
import argparse # Import the argparse library

def main():
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description='Analyze AlphaFold prediction results and rank peptides by inter-chain PAE.')
    parser.add_argument('--results_dir', required=True, help='The directory where your prediction files are stored.')
    parser.add_argument('--file_prefix', required=True, help='The prefix you used for your output files.')
    parser.add_argument('--output_csv', required=True, help='Path to save the final ranked CSV file.')
    args = parser.parse_args()

    # --- Main Analysis ---
    print(f"Analyzing results in: {args.results_dir}")

    results = []
    pae_files = glob.glob(os.path.join(args.results_dir, f'{args.file_prefix}_*_predicted_aligned_error.npy'))

    if not pae_files:
        print("Error: No PAE files found! Check your --results_dir and --file_prefix.")
        return
    else:
        print(f"Found {len(pae_files)} prediction files to analyze.")

    for pae_file_path in pae_files:
        match = re.search(f'{args.file_prefix}_[0-9]+_([A-Z]+)_', pae_file_path)
        if not match:
            continue
        peptide_sequence = match.group(1)

        pae_matrix = np.load(pae_file_path)

        # Using the correct protein length for this specific target set
        protein_len = 175
        peptide_len = 10
        total_len = pae_matrix.shape[0]

        if total_len != (protein_len + peptide_len):
            print(f"Warning: Skipping {peptide_sequence} due to unexpected length ({total_len})")
            continue

        pae_protein_vs_peptide = pae_matrix[0:protein_len, protein_len:]
        pae_peptide_vs_protein = pae_matrix[protein_len:, 0:protein_len]

        mean_pae_1 = np.mean(pae_protein_vs_peptide)
        mean_pae_2 = np.mean(pae_peptide_vs_protein)
        inter_chain_pae = (mean_pae_1 + mean_pae_2) / 2

        results.append({
            'peptide': peptide_sequence,
            'inter_chain_pae': inter_chain_pae
        })

    if results:
        results_df = pd.DataFrame(results)
        results_df_sorted = results_df.sort_values(by='inter_chain_pae', ascending=True)

        print("\n--- Top 10 Predicted Binders (Lower PAE is better) ---")
        print(results_df_sorted.head(10).to_string(index=False))

        results_df_sorted.to_csv(args.output_csv, index=False)
        print(f"\nFull ranked results saved to: {args.output_csv}")

if __name__ == '__main__':
    main()