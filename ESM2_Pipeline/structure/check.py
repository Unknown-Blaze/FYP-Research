import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# CONFIG
OUTPUT_DIR = Path("data/generated_pdbs/complexes")

def analyze_quality():
    # Only verify the files you just repaired
    pdb_files = list(OUTPUT_DIR.glob("complex_*.pdb"))
    
    if not pdb_files:
        print("No 'complex_*.pdb' files found.")
        return

    print(f"Found {len(pdb_files)} repaired files. Checking sample...")
    
    peptide_plddts = []
    
    # Shuffle and pick 2000 random files to test
    random.shuffle(pdb_files)
    sample_files = pdb_files[:2000]

    for pdb in tqdm(sample_files):
        plddt_scores = []
        has_chain_b = False
        
        try:
            with open(pdb, 'r') as f:
                for line in f:
                    if line.startswith("ATOM"):
                        # NOW we check standard column 21 (index 21) because you repaired it!
                        if len(line) > 21 and line[21] == 'B':
                            has_chain_b = True
                            try:
                                # Columns 60-66 are B-factor
                                score = float(line[60:66].strip())
                                plddt_scores.append(score)
                            except ValueError:
                                pass
        except Exception:
            continue
            
        if has_chain_b and plddt_scores:
            peptide_plddts.append(np.mean(plddt_scores))

    if not peptide_plddts:
        print("\n[WARNING] Scanned 2000 files but found NO Chain B.")
        print("These might be single-chain structures named 'complex_'. Check one manually.")
        return

    scores = np.array(peptide_plddts)
    print(f"\nSuccessfully analyzed {len(scores)} valid complexes.")
    print(f"Mean Confidence: {np.mean(scores):.2f}")
    
    # Histogram
    print("\nDistribution:")
    hist, _ = np.histogram(scores, bins=[0,50,70,80,90,100])
    labels = ["<50 (Junk)", "50-70 (Low)", "70-80 (Mid)", "80-90 (Good)", "90+ (High)"]
    for count, label in zip(hist, labels):
        bar = "#" * int(count / len(scores) * 50)
        print(f"{label:12} | {bar} ({count})")

if __name__ == "__main__":
    analyze_quality()