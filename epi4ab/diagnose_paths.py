import os
import pandas as pd
import argparse

def diagnose_paths(processed_dir, pdb_dir, sabdab_summary_path, num_to_check=20):
    """
    Checks if the paths and file lookups for the CDR script are correct.
    """
    print("--- Starting Diagnosis ---")
    
    # 1. Check if the main files and directories exist
    if not os.path.isdir(processed_dir):
        print(f"FATAL ERROR: Processed directory not found at '{processed_dir}'")
        return
    if not os.path.isdir(pdb_dir):
        print(f"FATAL ERROR: Raw PDB directory not found at '{pdb_dir}'")
        return
    if not os.path.exists(sabdab_summary_path):
        print(f"FATAL ERROR: SAbDab summary file not found at '{sabdab_summary_path}'")
        return
        
    print(f"Successfully found all input directories and files.")
    
    print("\nLoading SAbDab summary file...")
    try:
        sabdab_df = pd.read_csv(sabdab_summary_path, sep='\t', low_memory=False)
        print(f"Successfully loaded summary file with {len(sabdab_df)} rows.")
    except Exception as e:
        print(f"FATAL ERROR: Could not read the SAbDab summary file. Error: {e}")
        return

    pdb_dirs = sorted([d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))])
    print(f"Found {len(pdb_dirs)} processed PDB folders in '{processed_dir}'.")
    
    print(f"\n--- Checking a sample of the first {num_to_check} PDBs ---")
    
    issues_found = 0
    for pdb_id in pdb_dirs[:num_to_check]:
        print(f"\nCHECKING PDB ID: {pdb_id}")
        
        # Check 1: Is the PDB ID in the summary file?
        entry = sabdab_df[sabdab_df['pdb'].str.lower() == pdb_id.lower()]
        if entry.empty:
            print(f"  - [FAIL] PDB ID '{pdb_id}' was NOT FOUND in '{os.path.basename(sabdab_summary_path)}'.")
            issues_found += 1
        else:
            print(f"  - [OK] PDB ID '{pdb_id}' found in summary file.")

        # Check 2: Does the raw PDB file exist at the expected path?
        expected_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(expected_path):
            print(f"  - [FAIL] Raw PDB file was NOT FOUND at the path: '{expected_path}'.")
            issues_found += 1
        else:
            print(f"  - [OK] Raw PDB file found at the path: '{expected_path}'.")

    print("\n--- Diagnosis Complete ---")
    if issues_found == 0:
        print("All checked paths and lookups are correct. The issue is likely with ANARCI or PDB parsing.")
    else:
        print(f"Found {issues_found} issues. Please check the [FAIL] messages above.")
        print("The most common cause of failure is an incorrect path provided to the --pdb_dir argument.")
        print("Please verify that this path points to the directory containing your raw .pdb files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose path and file issues for the CDR generation script.")
    parser.add_argument("--processed_dir", type=str, required=True, help="e.g., data/nodes_edges/")
    parser.add_argument("--pdb_dir", type=str, required=True, help="e.g., data/raw_pdb/")
    parser.add_argument("--sabdab_summary", type=str, required=True, help="e.g., data/sabdab_summary_master.tsv")
    args = parser.parse_args()
    diagnose_paths(args.processed_dir, args.pdb_dir, args.sabdab_summary)
