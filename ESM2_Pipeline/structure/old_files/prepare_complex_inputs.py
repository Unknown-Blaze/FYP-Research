import pandas as pd
import hashlib
import json

# CONFIG
CSV_FILE = "../iedb_dataset.csv"
COMPLEX_FASTA_FILE = "complexes.fasta"
MAPPING_FILE = "row_to_pdb_map.json"

print(f"Loading {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"❌ Error: Could not find {CSV_FILE}")
    exit(1)

# Ensure we have the necessary columns
# Note: 'ic50_value' might be named 'IC50', 'ic50', or 'binding_affinity_nM' in your CSV.
# We check a few common names.
possible_labels = ['IC50', 'ic50_value', 'binding_affinity_nM', 'ic50']
label_col = next((c for c in possible_labels if c in df.columns), None)

df = df.dropna(subset=['mhc_sequence', 'peptide'])

print(f"Found {len(df)} rows. Using label column: {label_col}")

# --- Helper function to create a unique ID for the PAIR ---
def get_pair_id(mhc_seq, pep_seq):
    # Hash the COMBINATION of both sequences
    # We use a separator to ensure "ABC" + "DE" is distinct from "AB" + "CDE"
    combo = f"{mhc_seq}:{pep_seq}"
    return hashlib.md5(combo.encode()).hexdigest()

# 1. Process Unique Pairs
unique_pairs = {}
row_map = {}

print("Processing unique MHC:Peptide complexes...")

# We iterate through the dataframe to build the map AND find unique complexes simultaneously
for index, row in df.iterrows():
    mhc_seq = row['mhc_sequence'].strip().upper()
    pep_seq = row['peptide'].strip().upper()
    
    # Validation: Skip sequences that are too short to be realistic
    if len(mhc_seq) < 20 or len(pep_seq) < 3:
        continue

    # Generate the ID for this specific interaction
    pair_id = f"complex_{get_pair_id(mhc_seq, pep_seq)}"
    
    # Store in unique dictionary for FASTA generation
    # The ':' tells ColabFold these are two chains in one structure
    unique_pairs[pair_id] = f"{mhc_seq}:{pep_seq}"
    
    # Map the CSV row index to this structure ID
    row_data = {
        "structure_id": pair_id,
        "mhc_len": len(mhc_seq),
        "pep_len": len(pep_seq)
    }
    
    # Preserve the label if it exists
    if label_col:
        row_data["ic50"] = row[label_col]
        
    row_map[index] = row_data

# 2. Write Unique Complexes FASTA
print(f"Writing {len(unique_pairs)} unique complexes to FASTA...")
with open(COMPLEX_FASTA_FILE, 'w') as f:
    for pair_id, sequence in unique_pairs.items():
        f.write(f">{pair_id}\n{sequence}\n")

# 3. Write Mapping File
print("Writing mapping file...")
with open(MAPPING_FILE, 'w') as f:
    json.dump(row_map, f, indent=2)

print(f"✓ Success! Created:\n  {COMPLEX_FASTA_FILE} ({len(unique_pairs)} sequences)\n  {MAPPING_FILE}")
print("Next step: Run 'generate_complexes.sh'")