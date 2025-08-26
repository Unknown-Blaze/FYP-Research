import os
import pandas as pd
from tqdm import tqdm
import argparse
import requests
import time

# These are the columns your training script expects
AMINO_ACID_ONEHOT_COLS = [
    "isALA", "isCYS", "isASP", "isGLU", "isPHE", "isGLY", "isHIS", 
    "isILE", "isLYS", "isLEU", "isMET", "isASN", "isPRO", "isGLN", 
    "isARG", "isSER", "isTHR", "isVAL", "isTRP", "isTYR"
]
V_HEAVY_ONEHOT_COLS = [
    "IGHV1", "IGHV3", "IGHV5", "IGHV4", "IGHV8", "IGHV7", "IGHV6", 
    "IGHV9", "IGHV14", "unknown_VH"
]
V_LIGHT_ONEHOT_COLS = [
    "IGKV1", "IGLV2", "IGLV3", "IGKV3", "IGKV12", "IGKV4", "IGLV1", 
    "IGKV5", "IGKV8", "IGKV10", "IGKV6", "IGKV1D", "IGKV16", "unknown_VL"
]

def get_v_gene_from_api(pdb_id):
    """
    Queries the SAbDab API to get detailed info for a single PDB.
    """
    url = f"https://opig.stats.ox.ac.uk/sabdab/api/pdb/{pdb_id.lower()}/"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            h_gene = data.get('H_gene')
            l_gene = data.get('L_gene')
            
            vh_family = h_gene.split('-')[0].split('*')[0] if h_gene else None
            vl_family = l_gene.split('-')[0].split('*')[0] if l_gene else None
            
            return vh_family, vl_family
        else:
            # print(f"Warning: API call for {pdb_id} failed with status {response.status_code}")
            return None, None
    except requests.exceptions.RequestException as e:
        # print(f"Warning: API request for {pdb_id} failed: {e}")
        return None, None

def patch_parquet_file(file_path, vh_family, vl_family):
    """
    Loads a node_feature.parquet file, adds missing one-hot columns, and saves it.
    """
    df = pd.read_parquet(file_path)
    file_was_patched = False

    # Patch Amino Acid columns if they are missing
    if 'isALA' not in df.columns and 'resName' in df.columns:
        aa_dummies = pd.get_dummies(df['resName'], prefix='is')
        for col in AMINO_ACID_ONEHOT_COLS:
            if col not in aa_dummies.columns:
                aa_dummies[col] = 0
        df = pd.concat([df, aa_dummies[AMINO_ACID_ONEHOT_COLS]], axis=1)
        file_was_patched = True

    # Patch V-Gene columns if they are missing
    if 'IGHV1' not in df.columns:
        for col in V_HEAVY_ONEHOT_COLS + V_LIGHT_ONEHOT_COLS:
            df[col] = 0
        
        if vh_family and vh_family in df.columns:
            df[vh_family] = 1
        else:
            df['unknown_VH'] = 1
            
        if vl_family and vl_family in df.columns:
            df[vl_family] = 1
        else:
            df['unknown_VL'] = 1
        file_was_patched = True

    if file_was_patched:
        df.to_parquet(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update .parquet files with features fetched from the SAbDab API.")
    parser.add_argument("--processed_dir", type=str, required=True, help="Directory containing the processed PDB folders (e.g., data/nodes_edges).")
    args = parser.parse_args()

    pdb_dirs = [d for d in os.listdir(args.processed_dir) if os.path.isdir(os.path.join(args.processed_dir, d))]
    
    print(f"Found {len(pdb_dirs)} processed PDBs to patch using the SAbDab API.")
    
    for pdb_id in tqdm(pdb_dirs, desc="Patching Features via API"):
        node_file = os.path.join(args.processed_dir, pdb_id, 'node_feature.parquet')
        
        if not os.path.exists(node_file):
            continue
            
        vh, vl = get_v_gene_from_api(pdb_id)
        patch_parquet_file(node_file, vh, vl)
        
        # Be polite to the server by waiting a tiny bit between requests
        time.sleep(0.05)

    print("\nFeature patching complete.")