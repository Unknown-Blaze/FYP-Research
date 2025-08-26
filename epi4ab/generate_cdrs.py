import os
import pandas as pd
import json
import subprocess
import tempfile
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data import IUPACData
from tqdm import tqdm
import warnings
import argparse

# Suppress PDBConstructionWarning from BioPython
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

# --- Helper Dictionaries ---
THREE_TO_ONE_MAP = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}

def get_sabdab_data_from_df(pdb_id, sabdab_df):
    """Looks up Hchain and Lchain IDs for a single PDB from the provided DataFrame."""
    entry = sabdab_df[sabdab_df['pdb'].str.lower() == pdb_id.lower()]
    if entry.empty:
        return None
    return {'Hchain': entry.iloc[0]['Hchain'], 'Lchain': entry.iloc[0]['Lchain']}

def get_chain_sequence(chain):
    """Extracts amino acid sequence from a Bio.PDB chain object."""
    if not chain: return ""
    seq = ""
    for residue in chain.get_residues():
        if is_aa(residue, standard=True):
            seq += THREE_TO_ONE_MAP.get(residue.get_resname().upper(), 'X')
    return seq

def extract_cdrs_with_anarci(chain_id, sequence):
    """Runs ANARCI to get CDR sequences for a given chain sequence."""
    cdrs = {'cdr1_seq': '', 'cdr2_seq': '', 'cdr3_seq': ''}
    if not sequence:
        return cdrs

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".fasta") as tmp_fasta:
        tmp_fasta.write(f">{chain_id}\n{sequence}\n")
        fasta_path = tmp_fasta.name
    
    # --- CRITICAL FIX: Hardcode the paths to bypass shell environment issues ---
    python_interpreter = "/home3/rudrapra001/miniconda3/envs/epi4ab/bin/python"
    anarci_script = "/home3/rudrapra001/miniconda3/envs/epi4ab/bin/ANARCI"
    cmd = [python_interpreter, anarci_script, "-i", fasta_path, "-s", "imgt"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        # Silently fail if ANARCI has an issue with a specific sequence
        # For debugging, you could print the error: print(f"ANARCI failed for {chain_id}: {e.stderr}")
        os.unlink(fasta_path)
        return cdrs
    
    os.unlink(fasta_path)

    lines = result.stdout.splitlines()
    cdr1, cdr2, cdr3 = [], [], []
    cdr1_range, cdr2_range, cdr3_range = range(27, 39), range(56, 66), range(105, 118)

    for line in lines:
        parts = line.split()
        if line.startswith(('H', 'L')) and len(parts) >= 3 and parts[1].isdigit():
            pos, aa = int(parts[1]), parts[2]
            if aa == '-': continue
            if pos in cdr1_range: cdr1.append(aa)
            if pos in cdr2_range: cdr2.append(aa)
            if pos in cdr3_range: cdr3.append(aa)
            
    cdrs['cdr1_seq'] = "".join(cdr1)
    cdrs['cdr2_seq'] = "".join(cdr2)
    cdrs['cdr3_seq'] = "".join(cdr3)
    return cdrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CDR sequences for existing processed PDBs using a local SAbDab summary file.")
    parser.add_argument("--processed_dir", type=str, required=True, help="e.g., data/nodes_edges/")
    parser.add_argument("--pdb_dir", type=str, required=True, help="e.g., data/raw_pdb/")
    parser.add_argument("--sabdab_summary", type=str, required=True, help="e.g., data/sabdab_summary_master.tsv")
    args = parser.parse_args()

    print("Loading SAbDab summary file...")
    sabdab_df = pd.read_csv(args.sabdab_summary, sep='\t', low_memory=False)
    
    pdb_dirs = sorted([d for d in os.listdir(args.processed_dir) if os.path.isdir(os.path.join(args.processed_dir, d))])
    
    pdb_parser = PDBParser(PERMISSIVE=True)
    
    print(f"Found {len(pdb_dirs)} processed PDBs. Checking for missing CDRs...")
    
    successful_updates = 0
    # Process all PDBs with a progress bar
    for pdb_id in tqdm(pdb_dirs, desc="Generating CDRs"):
        seq_dir = os.path.join(args.processed_dir, pdb_id, 'sequence')
        cdr_file = os.path.join(seq_dir, 'cdr_sequence.json')

        # Skip if a valid CDR file already exists
        if os.path.exists(cdr_file):
            try:
                with open(cdr_file, 'r') as f: content = json.load(f)
                if "note" not in content and any(content.values()):
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

        sabdab_info = get_sabdab_data_from_df(pdb_id, sabdab_df)
        if not sabdab_info:
            continue

        pdb_file_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_file_path):
            continue
        
        try:
            structure = pdb_parser.get_structure(pdb_id, pdb_file_path)
            model = structure[0]
        except Exception:
            # Skip PDBs that Bio.PDB cannot parse
            continue

        h_chain_id = sabdab_info['Hchain'].split('/')[0] if pd.notna(sabdab_info['Hchain']) else None
        l_chain_id = sabdab_info['Lchain'].split('/')[0] if pd.notna(sabdab_info['Lchain']) else None
        
        h_chain = model[h_chain_id] if h_chain_id and h_chain_id in model else None
        l_chain = model[l_chain_id] if l_chain_id and l_chain_id in model else None

        h_chain_seq = get_chain_sequence(h_chain)
        l_chain_seq = get_chain_sequence(l_chain)

        h_cdrs = extract_cdrs_with_anarci(h_chain_id, h_chain_seq)
        l_cdrs = extract_cdrs_with_anarci(l_chain_id, l_chain_seq)

        final_cdr_data = {
            "H1_seq": h_cdrs.get('cdr1_seq', ''), "H2_seq": h_cdrs.get('cdr2_seq', ''), "H3_seq": h_cdrs.get('cdr3_seq', ''),
            "L1_seq": l_cdrs.get('cdr1_seq', ''), "L2_seq": l_cdrs.get('cdr2_seq', ''), "L3_seq": l_cdrs.get('cdr3_seq', '')
        }
        
        if any(final_cdr_data.values()):
            os.makedirs(seq_dir, exist_ok=True)
            with open(cdr_file, 'w') as f:
                json.dump(final_cdr_data, f, indent=4)
            successful_updates += 1

    print(f"\nCDR generation complete. Successfully updated {successful_updates} files.")
