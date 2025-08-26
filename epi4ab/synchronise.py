import pandas as pd

def synchronize():
    try:
        # Load the list of PDBs that actually have processed data
        with open('existing_pdbs.txt', 'r') as f:
            existing_pdbs = {line.strip() for line in f}
        print(f"Found {len(existing_pdbs)} PDBs with existing data folders.")

        # Load the original train and test lists
        original_train_df = pd.read_csv('lists/train_pdb.txt', header=None, names=['pdb'])
        original_test_df = pd.read_csv('lists/test_pdb.txt', header=None, names=['pdb'])
        
        # Find the intersection - PDBs that are in the lists AND have data
        synced_train_pdbs = [pdb for pdb in original_train_df['pdb'] if pdb in existing_pdbs]
        synced_test_pdbs = [pdb for pdb in original_test_df['pdb'] if pdb in existing_pdbs]

        print(f"Original train list had {len(original_train_df)} entries. Synced version has {len(synced_train_pdbs)}.")
        print(f"Original test list had {len(original_test_df)} entries. Synced version has {len(synced_test_pdbs)}.")

        # Save the new, synchronized lists
        pd.DataFrame(synced_train_pdbs).to_csv('lists/train_pdb_synced.txt', index=False, header=False)
        pd.DataFrame(synced_test_pdbs).to_csv('lists/test_pdb_synced.txt', index=False, header=False)

        print("\nSuccessfully created 'train_pdb_synced.txt' and 'test_pdb_synced.txt'.")
        print("Please update your .env file to use these new files.")

    except FileNotFoundError as e:
        print(f"Error: Make sure you have run 'ls -1 data/nodes_edges/ > existing_pdbs.txt' first.")
        print(f"Missing file: {e.filename}")

if __name__ == "__main__":
    synchronize()
