# RUN THIS ONLY IF YOU RESTART AFTER A CRASH/STOP
from pathlib import Path
output_dir = Path("data/generated_pdbs/complexes")
for f in output_dir.glob("*.pdb"):
    if f.stat().st_size < 1000: # If file is suspiciously small (<1KB)
        print(f"Deleting incomplete file: {f.name}")
        f.unlink()