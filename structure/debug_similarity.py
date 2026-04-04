import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import pandas as pd
import difflib
import os
import warnings

warnings.filterwarnings("ignore")

# CONFIG
TEMPLATE_FILE = "templates/template_1hhi.pdb" # Let's test just one
CSV_FILE = "../iedb_dataset.csv"

def check_similarity():
    # 1. Load CSV Target
    print("--- 1. LOADING TARGET (CSV) ---")
    df = pd.read_csv(CSV_FILE)
    target_seq = df.iloc[0]['mhc_sequence'].upper()
    print(f"Target Length: {len(target_seq)}")
    print(f"Target Preview: {target_seq[:50]}...")

    # 2. Load Template (PDB)
    print("\n--- 2. LOADING TEMPLATE (PDB) ---")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("debug", TEMPLATE_FILE)
    
    found_seq = ""
    
    for chain in structure[0]:
        residues = [res for res in chain if 'CA' in res and res.id[0] == ' ']
        length = len(residues)
        
        # Try to convert 3-letter to 1-letter
        try:
            # Custom 3-to-1 map fallback if seq1 fails
            d3to1 = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G',
                     'HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N',
                     'PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V',
                     'TRP':'W','TYR':'Y'}
            
            # Try Biopython seq1 first, fallback to manual dict
            seq_list = []
            for res in residues:
                rname = res.get_resname().strip().upper()
                # Handle non-standard residues often found in PDBs
                if rname in d3to1:
                    seq_list.append(d3to1[rname])
                else:
                    seq_list.append('X') # Mark unknown as X
            
            sequence = "".join(seq_list)
            
        except Exception as e:
            sequence = "ERROR"

        print(f"Chain {chain.get_id()}: {length} residues")
        if length > 50:
            print(f"   Seq Preview: {sequence[:50]}...")
        
        if length > 150: # Assume this is the MHC
            found_seq = sequence

    # 3. Compare
    print("\n--- 3. COMPARISON ---")
    if found_seq:
        ratio = difflib.SequenceMatcher(None, target_seq, found_seq).ratio()
        print(f"Calculated Identity: {ratio:.4f} ({ratio*100:.2f}%)")
        
        # Show alignment block
        matcher = difflib.SequenceMatcher(None, target_seq, found_seq)
        for block in matcher.get_matching_blocks():
            if block.size > 5:
                print(f"MATCH BLOCK: Start Target={block.a}, Start Template={block.b}, Length={block.size}")
                print(f"   Segment: {target_seq[block.a : block.a + block.size]}")
    else:
        print("ERROR: Could not isolate an MHC sequence from the PDB.")

if __name__ == "__main__":
    if os.path.exists(TEMPLATE_FILE) and os.path.exists(CSV_FILE):
        check_similarity()
    else:
        print("Files not found. Check paths.")