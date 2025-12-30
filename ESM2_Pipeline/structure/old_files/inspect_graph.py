import torch
from pathlib import Path
from tqdm import tqdm
import os

# CONFIG
GRAPHS_DIR = Path("data/graphs/ensemble/mhc")

def check_files():
    if not GRAPHS_DIR.exists():
        print(f"Directory not found: {GRAPHS_DIR}")
        return

    files = list(GRAPHS_DIR.glob("*.pt"))
    total = len(files)
    print(f"Found {total} graph files. Scanning for 'identity' attribute...")

    missing_count = 0
    present_count = 0
    corrupted_count = 0

    # Scan a subset if too many, or all if you want to be sure
    # Using tqdm for progress bar
    for f in tqdm(files):
        try:
            data = torch.load(f)
            if hasattr(data, 'identity'):
                present_count += 1
            else:
                missing_count += 1
                # Optional: Print the first missing one to debug
                if missing_count == 1:
                    print(f"\n[EXAMPLE] File missing identity: {f.name}")
        except Exception:
            corrupted_count += 1

    print("\n" + "="*30)
    print(f"REPORT:")
    print(f"Total Files:     {total}")
    print(f"✅ Good Files:    {present_count} (Have 'identity')")
    print(f"❌ Bad Files:     {missing_count} (Missing 'identity')")
    print(f"⚠️ Corrupted:     {corrupted_count}")
    print("="*30)

    if missing_count > 0:
        print("\nSUGGESTION: The bad files are likely leftovers from a previous run.")
        print("Safest solution: Delete the entire 'data/graphs' folder and re-run generation.")
        print("Quickest solution: Use the 'Robust Dataset' code I provided (it handles missing keys).")

if __name__ == "__main__":
    check_files()