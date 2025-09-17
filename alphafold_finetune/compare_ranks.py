import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Compare benchmark and fine-tuned model rankings.')
    parser.add_argument('--finetuned_csv', required=True, help='CSV file with rankings from the fine-tuned model.')
    parser.add_argument('--benchmark_csv', required=True, help='CSV file with rankings from the benchmark (default) model.')
    parser.add_argument('--output_csv', required=True, help='Path to save the final comparison CSV file.')
    args = parser.parse_args()

    # Load the two ranked lists
    ft_df = pd.read_csv(args.finetuned_csv)
    bm_df = pd.read_csv(args.benchmark_csv)

    # Add a 'rank' column to each DataFrame
    ft_df['rank_finetuned'] = ft_df['inter_chain_pae'].rank()
    bm_df['rank_benchmark'] = bm_df['inter_chain_pae'].rank()

    # Merge the two dataframes based on the peptide sequence
    comparison_df = pd.merge(
        ft_df,
        bm_df,
        on='peptide',
        suffixes=('_finetuned', '_benchmark')
    )

    # Calculate the improvement in rank
    # A positive number means the fine-tuned model ranked it better (closer to #1)
    comparison_df['rank_improvement'] = comparison_df['rank_benchmark'] - comparison_df['rank_finetuned']

    # Sort by the biggest improvement
    comparison_df_sorted = comparison_df.sort_values(by='rank_improvement', ascending=False)

    # Reorder columns for clarity
    final_df = comparison_df_sorted[[
        'peptide',
        'rank_finetuned',
        'rank_benchmark',
        'rank_improvement',
        'inter_chain_pae_finetuned',
        'inter_chain_pae_benchmark'
    ]]

    print("\n--- Peptides with the BIGGEST Improvement from Fine-Tuning ---")
    print(final_df.head(20).to_string(index=False))
    
    print("\n--- Peptides that got WORSE after Fine-Tuning ---")
    print(final_df.tail(10).to_string(index=False))

    # Save the full comparison to a file
    final_df.to_csv(args.output_csv, index=False)
    print(f"\nFull comparison table saved to: {args.output_csv}")


if __name__ == '__main__':
    main()