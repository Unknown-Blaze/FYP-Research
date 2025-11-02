# prepare_fasta.py
import csv
import sys
from pathlib import Path

# Input CSV file (relative path)
csv_file = Path(__file__).parent / "../iedb_dataset.csv"

# Output FASTA file
fasta_file = Path(__file__).parent / "colabfold_input.fasta"

# Columns from your script
mhc_col = "mhc_sequence"
pep_col = "peptide"
mhc_id_col = "mhc_allele"

sequences_to_write = []
try:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        MAX_SAMPLES = 10 # <--- ADD THIS LINE
        
        for i, row in enumerate(reader):
        
            if i >= MAX_SAMPLES: # <--- ADD THIS LINE
                break            # <--- AND THIS ONE

            mhc_seq = str(row.get(mhc_col, "")).strip().upper()
            pep_seq = str(row.get(pep_col, "")).strip().upper()
            
            if not mhc_seq or not pep_seq:
                continue
                
            # Create a unique, clean ID
            mhc_id = str(row.get(mhc_id_col, "MHC")).split("*")[-1][:10]
            pep_id = str(len(pep_seq))
            header = f"{i}_{mhc_id}_{pep_id}"
            
            # Format for ColabFold complex: SEQ1:SEQ2
            combined_seq = f"{mhc_seq}:{pep_seq}"
            
            sequences_to_write.append(f">{header}\n{combined_seq}\n")

    with open(fasta_file, 'w') as f:
        f.writelines(sequences_to_write)
        
    print(f"✓ Successfully created {fasta_file.name} with {len(sequences_to_write)} sequences.")

except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)