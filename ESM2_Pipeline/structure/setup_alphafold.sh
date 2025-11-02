#!/bin/bash
# Find where AlphaFold is installed and generate structures

echo "Searching for AlphaFold installation..."

# Check common locations
AF_LOCATIONS=(
    "$HOME/alphafold_finetune"
    "$HOME/alphafold"
    "/opt/alphafold"
    "/home/alphafold"
    "$(which run_alphafold.sh 2>/dev/null | xargs dirname)"
)

AF_PATH=""
for loc in "${AF_LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "✓ Found AlphaFold at: $loc"
        AF_PATH="$loc"
        break
    fi
done

if [ -z "$AF_PATH" ]; then
    echo "✗ AlphaFold not found in common locations"
    echo ""
    echo "Please tell me where AlphaFold is installed:"
    read -p "Enter AlphaFold path: " AF_PATH
    
    if [ ! -d "$AF_PATH" ]; then
        echo "✗ Directory not found: $AF_PATH"
        exit 1
    fi
fi

echo "Using AlphaFold at: $AF_PATH"
echo ""

# Create output directory
mkdir -p ./data/generated_pdbs

# Activate AlphaFold environment if it exists
if [ -f "$AF_PATH/env.sh" ]; then
    echo "Activating AlphaFold environment..."
    source "$AF_PATH/env.sh"
fi

# Run AlphaFold for first 10 samples
echo "Generating MHC-Peptide structures..."
echo ""

# Create FASTA file from dataset
python3 << 'EOF'
import pandas as pd
import os

# Load dataset
df = pd.read_csv("../iedb_dataset.csv", on_bad_lines='skip')
df = df.head(10)  # First 10 samples

# Create FASTA
fasta_file = "./data/generated_pdbs/sequences.fasta"
with open(fasta_file, 'w') as f:
    for idx, row in df.iterrows():
        mhc_seq = str(row['mhc_sequence']).strip()
        pep_seq = str(row['peptide']).strip()
        
        # ColabFold format: separate by colon
        f.write(f">complex_{idx:06d}\n{mhc_seq}:{pep_seq}\n")

print(f"✓ Created FASTA: {fasta_file}")
print(f"  Samples: {len(df)}")
print(f"  Example sequence: {df.iloc[0]['mhc_sequence'][:30]}...:{df.iloc[0]['peptide']}")
EOF

echo ""
echo "=========================================="
echo "FASTA file ready: ./data/generated_pdbs/sequences.fasta"
echo "=========================================="
echo ""
echo "Next steps to generate structures:"
echo ""
echo "OPTION 1 - ColabFold Web (Easiest, Fastest):"
echo "  1. Go to: https://colabfold.mmseqs.com/"
echo "  2. Upload: ./data/generated_pdbs/sequences.fasta"
echo "  3. Download result.zip"
echo "  4. Extract *.pdb files to: ./data/generated_pdbs/"
echo ""
echo "OPTION 2 - Local AlphaFold (if you know how to run it):"
echo "  Run your local AlphaFold installation with:"
echo "  - Input: ./data/generated_pdbs/sequences.fasta"
echo "  - Model: multimer (for MHC-peptide complexes)"
echo "  - Output: ./data/generated_pdbs/"
echo ""
echo "Then run: python 2_build_graphs_simple.py"
echo ""