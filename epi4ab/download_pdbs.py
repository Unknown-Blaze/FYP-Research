# download_pdbs.py
import os
import requests
from tqdm import tqdm

# --- Configuration ---
LISTS_DIR = "./lists"
TRAIN_FILE = os.path.join(LISTS_DIR, "train_pdb.txt")
TEST_FILE = os.path.join(LISTS_DIR, "test_pdb.txt")
OUTPUT_DIR = "./data/raw_pdb"
# ---------------------

def get_pdb_ids():
    """Reads PDB IDs from train and test lists."""
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print(f"Error: Make sure {TRAIN_FILE} and {TEST_FILE} exist.")
        return set()
        
    with open(TRAIN_FILE, 'r') as f:
        train_ids = {line.strip() for line in f if line.strip()}
    with open(TEST_FILE, 'r') as f:
        test_ids = {line.strip() for line in f if line.strip()}
        
    all_ids = train_ids.union(test_ids)
    print(f"Found {len(all_ids)} unique PDB IDs to download.")
    return all_ids

def download_pdb(pdb_id, output_dir):
    """Downloads a single PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    filepath = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(filepath):
        # print(f"Skipping {pdb_id}, already exists.")
        return True

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        
        with open(filepath, 'w') as f:
            f.write(response.text)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {pdb_id}. Error: {e}")
        return False

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pdb_ids = get_pdb_ids()
    
    if pdb_ids:
        print(f"Downloading PDB files to {OUTPUT_DIR}...")
        # Use tqdm for a progress bar
        success_count = 0
        for pdb_id in tqdm(pdb_ids, desc="Downloading PDBs"):
            if download_pdb(pdb_id, OUTPUT_DIR):
                success_count += 1
        
        print(f"\nDownload complete. Successfully downloaded {success_count}/{len(pdb_ids)} files.")
