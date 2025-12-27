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

# Human Beta-2 Microglobulin
B2M_SEQ = "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"

# GOLDEN STANDARD LINKER: (GGGGS)x3
# Stable enough to fold, flexible enough to dock.
LINKER_SEQ = "GGGGSGGGGSGGGGS" 
# ==========================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_structure_shape(positions):
    if hasattr(positions, 'cpu'):
        positions = positions.detach().cpu().float().numpy()    
    shape = positions.shape
    if len(shape) == 5: positions = positions[-1][0]
    elif len(shape) == 4: positions = positions[0]
    return positions

def save_reordered_pdb(coords, plddt_scores, full_seq, len_pep, len_b2m, len_mhc, linker_len, filename):
    """
    Input Order:  Peptide -> Linker -> B2M -> Linker -> MHC
    Output Order: Chain A (MHC), Chain B (Peptide), Chain C (B2M)
    """
    
    # Calculate RANGES in the folded sequence
    # 1. Peptide (Start)
    range_pep = (0, len_pep)
    
    # 2. B2M (Middle)
    start_b2m = len_pep + linker_len
    range_b2m = (start_b2m, start_b2m + len_b2m)
    
    # 3. MHC (End)
    start_mhc = start_b2m + len_b2m + linker_len
    range_mhc = (start_mhc, start_mhc + len_mhc)

    atom_indices = {'N': 0, 'CA': 1, 'C': 2, 'O': 3} if coords.shape[1] == 14 else {'N': 0, 'CA': 1, 'C': 2, 'O': 4}
    aa_map = {'A':'ALA', 'R':'ARG', 'N':'ASN', 'D':'ASP', 'C':'CYS', 'Q':'GLN', 'E':'GLU', 'G':'GLY', 'H':'HIS', 'I':'ILE', 'L':'LEU', 'K':'LYS', 'M':'MET', 'F':'PHE', 'P':'PRO', 'S':'SER', 'T':'THR', 'W':'TRP', 'Y':'TYR', 'V':'VAL', 'X':'UNK'}

    with open(filename, 'w') as f:
        atom_serial = 1
        
        def write_segment(start_idx, end_idx, chain_id):
            nonlocal atom_serial
            for i in range(start_idx, end_idx):
                if i >= coords.shape[0]: break
                
                aa = full_seq[i]
                res_name = aa_map.get(aa, 'UNK')
                # Residue ID restarts at 1 for each chain
                res_id = (i - start_idx) + 1
                
                raw_score = float(plddt_scores[i, 1]) if plddt_scores.ndim == 2 else float(plddt_scores[i])
                b_factor = raw_score * 100.0

                for atom_name in ['N', 'CA', 'C', 'O']:
                    col_idx = atom_indices.get(atom_name)
                    if col_idx >= coords.shape[1]: continue
                    pos = coords[i, col_idx]
                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                    if abs(x) < 0.001 and abs(y) < 0.001: continue
                    
                    f.write(f"ATOM  {atom_serial:5d}  {atom_name:<4} {res_name}{chain_id}{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}           {atom_name[0]}\n")
                    atom_serial += 1

        # --- WRITE IN STANDARD PDB ORDER ---
        
        # 1. Chain A: MHC (From end of tensor)
        write_segment(range_mhc[0], range_mhc[1], 'A')
        
        # 2. Chain B: Peptide (From start of tensor)
        write_segment(range_pep[0], range_pep[1], 'B')
        
        # 3. Chain C: B2M (From middle of tensor)
        write_segment(range_b2m[0], range_b2m[1], 'C')
        
        f.write("TER\n")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading ESMFold...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loading Dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['mhc_sequence', 'peptide'])
    
    def get_id(r): return hashlib.md5(f"{r['mhc_sequence']}:{r['peptide']}".encode()).hexdigest()[:8]
    df['complex_id'] = df.apply(get_id, axis=1)
    unique_df = df.drop_duplicates(subset=['complex_id'])
    
    existing_ids = {p.stem.split('_')[-1] for p in OUTPUT_DIR.glob("*.pdb")}
    to_process = unique_df[~unique_df['complex_id'].isin(existing_ids)]
    logger.info(f"Remaining: {len(to_process)}")
        
    for idx, row in tqdm(to_process.iterrows(), total=len(to_process)):
        mhc = row['mhc_sequence'].strip().upper()
        pep = row['peptide'].strip().upper()
        cid = row['complex_id']
        
        # --- SCT TOPOLOGY ---
        # Lab standard: Pep -> Linker -> B2M -> Linker -> MHC
        full_seq = pep + LINKER_SEQ + B2M_SEQ + LINKER_SEQ + mhc
        
        len_pep = len(pep)
        len_b2m = len(B2M_SEQ)
        len_mhc = len(mhc)
        len_linker = len(LINKER_SEQ)

        # CONTINUOUS POSITIONS
        position_ids = torch.arange(len(full_seq), dtype=torch.long).unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                tokenized = tokenizer(full_seq, return_tensors="pt", add_special_tokens=False)
                input_ids = tokenized['input_ids'].to(device)
                
                output = model(input_ids, position_ids=position_ids)
                
                cleaned_coords = solve_structure_shape(output.positions)
                plddt = output.plddt.detach().cpu().float().numpy()[0]

            out_path = OUTPUT_DIR / f"complex_{cid}.pdb"
            
            # Save while unscrambling the order
            save_reordered_pdb(
                cleaned_coords, plddt, full_seq, 
                len_pep, len_b2m, len_mhc, len_linker, 
                out_path
            )
            
        except Exception as e:
            logger.error(f"Failed {cid}: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()