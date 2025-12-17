import sys
from pathlib import Path
import json
import re

BASE = Path(__file__).parent
MHC_DIR = BASE / 'data/generated_pdbs/mhc'
PEP_DIR = BASE / 'data/generated_pdbs/peptide'


def read_plddt_from_pdb(pdb_path: Path):
    # AlphaFold writes pLDDT into B-factor column (columns 61-66)
    plddts = []
    try:
        with pdb_path.open('r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        # B-factor at columns 61-66 (1-based), slice [60:66)
                        b = float(line[60:66].strip())
                        plddts.append(b)
                    except Exception:
                        pass
    except FileNotFoundError:
        return None
    if not plddts:
        return None
    import numpy as np
    return float(np.mean(plddts)), float(np.median(plddts)), float(min(plddts)), float(max(plddts))


def find_scores_json(pdb_path: Path):
    stem = pdb_path.name.split('.pdb')[0]
    # Remove trailing specific suffix like _unrelaxed_rank_001_...
    # Use shared prefix up to the last underscore before 'rank_'
    m = re.search(r'(.+?)_rank_\d{3}', stem)
    prefix = m.group(1) if m else stem
    # Look for any *_scores_*.json nearby
    for cand in pdb_path.parent.glob(f"{prefix}*scores*.json"):
        return cand
    return None


def read_scores(scores_path: Path):
    try:
        with scores_path.open('r') as f:
            data = json.load(f)
        # Common keys: ptm, iptm, mean_plddt, max_pae
        return {
            'ptm': data.get('ptm'),
            'iptm': data.get('iptm'),
            'mean_plddt': data.get('mean_plddt'),
            'max_pae': data.get('max_pae')
        }
    except Exception:
        return None


def summarize(dir_path: Path, label: str):
    pdbs = sorted(dir_path.glob('*rank_001*.pdb'))
    print(f"\n[{label}] Found {len(pdbs)} rank_001 PDBs in {dir_path}")
    if not pdbs:
        return

    import numpy as np
    means = []
    medians = []
    mins = []
    maxs = []

    for pdb in pdbs:
        stats = read_plddt_from_pdb(pdb)
        if stats is None:
            continue
        mean_plddt, median_plddt, min_plddt, max_plddt = stats
        means.append(mean_plddt)
        medians.append(median_plddt)
        mins.append(min_plddt)
        maxs.append(max_plddt)

    if not means:
        print("  No pLDDT found in PDB B-factors.")
        return

    print(f"  pLDDT mean (avg across structures): {np.mean(means):.1f}")
    print(f"  pLDDT median (avg across structures): {np.mean(medians):.1f}")
    print(f"  pLDDT min across all atoms: {np.min(mins):.1f}")
    print(f"  pLDDT max across all atoms: {np.max(maxs):.1f}")

    # Try to read the first structure's score JSON for ptm/iptm
    first_pdb = pdbs[0]
    sj = find_scores_json(first_pdb)
    if sj:
        scores = read_scores(sj)
        if scores:
            if scores.get('ptm') is not None:
                print(f"  pTM: {scores['ptm']:.3f}")
            if scores.get('iptm') is not None:
                print(f"  ipTM: {scores['iptm']:.3f}")
            if scores.get('mean_plddt') is not None:
                print(f"  mean_pLDDT (JSON): {scores['mean_plddt']:.1f}")


if __name__ == '__main__':
    print('AlphaFold/ColabFold structure quality summary (pLDDT from B-factors)')
    summarize(MHC_DIR, 'MHC')
    summarize(PEP_DIR, 'Peptides')
