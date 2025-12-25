import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import hashlib
import traceback

torch.set_float32_matmul_precision('high')

# CONFIG
# ==========================================================
INPUT_CSV = "../iedb_dataset.csv"
OUTPUT_DIR = Path("data/generated_pdbs/complexes")
# ==========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_structure_shape(positions):
    """
    Intelligently handles ESMFold output shapes (e.g., recycling steps, Atom14 vs Atom37).
    Target: Returns a numpy array of shape (SeqLen, Atoms, 3) representing the FINAL structure.
    """
    # Move to numpy if tensor
    if hasattr(positions, 'cpu'):
        positions = positions.detach().cpu().float().numpy()    
        
    shape = positions.shape
    
    # CASE 1: The "Recycling Stack" (8, 1, Seq, 14, 3)
    # We want the LAST recycle step (index -1), and the FIRST batch (index 0).
    if len(shape) == 5:
        # logging.debug(f"Detected 5D shape {shape}. Assuming (Recycles, Batch, Seq, Atoms, XYZ). Taking last recycle.")
        positions = positions[-1] # Take last recycle -> (Batch, Seq, Atoms, XYZ)
        if positions.shape[0] == 1:
            positions = positions[0] # Squeeze batch -> (Seq, Atoms, XYZ)

    # CASE 2: Standard Batch (1, Seq, 37, 3)
    elif len(shape) == 4:
        if shape[0] == 1:
            positions = positions[0] # Squeeze batch

    # Final Check
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"Could not resolve shape to (Seq, Atoms, 3). Got: {positions.shape}")
        
    return positions

# Updated function to handle pLDDT scores
def save_pdb_split_chains(coords, plddt_scores, sequence, mhc_len, pep_len, filename):
    """
    Saves PDB coords with pLDDT scores in the B-factor column.
    """
    
    # --- FIX IS HERE ---
    # Since full_seq = mhc + pep, the peptide starts immediately after the MHC.
    pep_start_idx = mhc_len 
    # -------------------
    
    atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3} if coords.shape[1] == 14 else {'N': 0, 'CA': 1, 'C': 2, 'O': 4}
    
    aa_map = {
        'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN', 
        'E':'GLU', 'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 
        'M':'MET', 'F':'PHE', 'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP', 
        'Y':'TYR', 'V':'VAL', 'X':'UNK'
    }

    with open(filename, 'w') as f:
        atom_serial = 1
        
        def write_residue(seq_idx, chain_id, res_id):
            nonlocal atom_serial
            
            # Safety check
            if seq_idx >= len(sequence) or seq_idx >= coords.shape[0]:
                return

            aa = sequence[seq_idx]
            res_name = aa_map.get(aa, 'UNK')
            
            # Extract pLDDT for this residue. 
            if plddt_scores.ndim == 2:
                raw_score = float(plddt_scores[seq_idx, 1]) 
            else:
                raw_score = float(plddt_scores[seq_idx])
            
            b_factor = raw_score * 100.0

            for atom_name in ['N', 'CA', 'C', 'O']:
                col_idx = atom_indices.get(atom_name)
                
                if col_idx >= coords.shape[1]: continue
                
                pos = coords[seq_idx, col_idx]
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                
                if abs(x) < 0.001 and abs(y) < 0.001 and abs(z) < 0.001: continue

                f.write(f"ATOM  {atom_serial:5d}  {atom_name:<4} {res_name}{chain_id}{res_id:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}           {atom_name[0]}\n")
                atom_serial += 1

        # --- CHAIN A (MHC) ---
        for i in range(mhc_len):
            write_residue(i, 'A', i + 1)

        # --- CHAIN B (Peptide) ---
        for i in range(pep_len):
            real_idx = pep_start_idx + i
            write_residue(real_idx, 'B', i + 1)
                
        f.write("TER\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading ESMFold Model (facebook/esmfold_v1)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)    
    model = model.to(device)
    model.eval()
    
    # if hasattr(torch, "compile"):
    #     logger.info("Compiling model for speed...")
    #     model = torch.compile(model)
    
    logger.info(f"Loading Dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['mhc_sequence', 'peptide'])
    
    # Generate IDs
    def get_id(r): return hashlib.md5(f"{r['mhc_sequence']}:{r['peptide']}".encode()).hexdigest()[:8]
    df['complex_id'] = df.apply(get_id, axis=1)
    
    unique_df = df.drop_duplicates(subset=['complex_id'])
    
    # Resume Logic
    existing_ids = {p.stem.split('_')[-1] for p in OUTPUT_DIR.glob("*.pdb")}
    to_process = unique_df[~unique_df['complex_id'].isin(existing_ids)]
    
    logger.info(f"Total Unique: {len(unique_df)} | Done: {len(existing_ids)} | Remaining: {len(to_process)}")
        
    # Loop
    for idx, row in tqdm(to_process.iterrows(), total=len(to_process)):
        mhc = row['mhc_sequence'].strip().upper()
        pep = row['peptide'].strip().upper()
        cid = row['complex_id']
        
        full_seq = mhc + pep
        # 2. Calculate lengths
        len_mhc = len(mhc)
        len_pep = len(pep)

        # 3. Create Position IDs (The "Chain Break")
        # MHC gets 0 to len_mhc
        mhc_pos = torch.arange(0, len_mhc, dtype=torch.long)
        # Peptide gets distinct positions far away (e.g., +200 offset)
        pep_pos = torch.arange(200 + len_mhc, 200 + len_mhc + len_pep, dtype=torch.long)

        # Concatenate
        position_ids = torch.cat([mhc_pos, pep_pos]).unsqueeze(0).to(device) # Shape: (1, SeqLen)

        try:
            with torch.no_grad():
                tokenized = tokenizer(full_seq, return_tensors="pt", add_special_tokens=False)
                input_ids = tokenized['input_ids'].to(device)
                
                # Pass position_ids to the model
                output = model(input_ids, position_ids=position_ids)
                
                cleaned_coords = solve_structure_shape(output.positions)
                
                # NEW: Extract pLDDT
                # Output pLDDT shape is usually (Batch, Seq, 37)
                plddt = output.plddt.detach().cpu().float().numpy()[0] # Take batch 0 -> (Seq, 37)

                out_path = OUTPUT_DIR / f"complex_{cid}.pdb"
                # Pass plddt to the save function
                save_pdb_split_chains(cleaned_coords, plddt, full_seq, len(mhc), len(pep), out_path)
            
        except IndexError as e:
            # THIS IS THE NEW BLOCK
            err_msg = str(e)
            if "index 0 is out of bounds" in err_msg:
                error_type = "NaN_Crash (TM-score)"
            else:
                error_type = "IndexError"

            with open("failed_sequences.log", "a") as logf:
                logf.write(f"ID: {cid}\n")
                logf.write(f"Error: {error_type}\n")
                logf.write(f"MHC Len: {len_mhc}, Pep Len: {len_pep}\n")
                logf.write(f"Max Pos ID: {position_ids.max().item()}\n")
                logf.write(f"MHC: {mhc}\n")
                logf.write(f"PEP: {pep}\n")
                logf.write("-" * 30 + "\n")
            
            logger.error(f"SKIPPING {cid} due to {error_type}. Logged to failed_sequences.log")
            
        except Exception as e:
            # Catch-all for other issues
            logger.error(f"Failed {cid}: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()