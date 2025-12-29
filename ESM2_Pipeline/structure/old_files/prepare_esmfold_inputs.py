# prepare_esmfold_inputs.py
import pandas as pd
import hashlib
import json

CSV_FILE = "../iedb_dataset.csv"
MHC_FASTA_FILE = "mhc_unique.fasta"
PEPTIDE_FASTA_FILE = "peptide_unique.fasta"
MAPPING_FILE = "row_to_pdb_map.json"

print(f"Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
df = df.dropna(subset=['mhc_sequence', 'peptide'])

# 1. Find all unique sequences
unique_mhc_seqs = {seq.strip().upper() for seq in df['mhc_sequence']}
unique_pep_seqs = {seq.strip().upper() for seq in df['peptide']}

print(f"Found {len(unique_mhc_seqs)} unique MHCs and {len(unique_pep_seqs)} unique peptides.")

# --- Helper function to create a unique, filesystem-safe ID ---
def get_seq_id(seq):
    # Create a stable hash for a given sequence
    return hashlib.md5(seq.encode()).hexdigest()

# 2. Write unique MHC FASTA
mhc_id_map = {}
with open(MHC_FASTA_FILE, 'w') as f:
    for seq in unique_mhc_seqs:
        if not seq: continue
        seq_id = f"mhc_{get_seq_id(seq)}"
        mhc_id_map[seq] = seq_id
        f.write(f">{seq_id}\n{seq}\n")

# 3. Write unique Peptide FASTA
pep_id_map = {}
with open(PEPTIDE_FASTA_FILE, 'w') as f:
    for seq in unique_pep_seqs:
        if not seq: continue
        seq_id = f"pep_{get_seq_id(seq)}"
        pep_id_map[seq] = seq_id
        f.write(f">{seq_id}\n{seq}\n")

# 4. Create the mapping file for your GNN
# This file links each row in your CSV to the PDB files you will generate
print("Creating mapping file for GNN...")
row_map = {}
for index, row in df.iterrows():
    mhc_seq = row['mhc_sequence'].strip().upper()
    pep_seq = row['peptide'].strip().upper()
    
    if mhc_seq in mhc_id_map and pep_seq in pep_id_map:
        row_map[index] = {
            "mhc_pdb_id": mhc_id_map[mhc_seq],
            "peptide_pdb_id": pep_id_map[pep_seq],
            "ic50": row.get("ic50_value") # Or whatever your column is
        }

with open(MAPPING_FILE, 'w') as f:
    json.dump(row_map, f, indent=2)

print(f"✓ Success! Created:\n  {MHC_FASTA_FILE}\n  {PEPTIDE_FASTA_FILE}\n  {MAPPING_FILE}")