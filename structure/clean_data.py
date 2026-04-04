import torch
import os
from pathlib import Path
from tqdm import tqdm

# CONFIG
TARGET_DIR = Path("data/graphs/ensemble/mhc")
REQUIRED_DIM = 31  # The new Physics dimension
OLD_DIM = 6        # The old sequence-only dimension

def clean_directory():
    if not TARGET_DIR.exists():
        print(f"Directory {TARGET_DIR} does not exist.")
        return

    files = list(TARGET_DIR.glob("*.pt"))
    print(f"Scanning {len(files)} files in {TARGET_DIR}...")

    deleted_count = 0
    kept_count = 0
    error_count = 0

    for file_path in tqdm(files, desc="Checking Dimensions"):
        try:
            # Load only the structure, map_location helps with speed/memory
            data = torch.load(file_path, map_location='cpu')
            
            # Check feature dimension
            current_dim = data.x.shape[1]
            
            if current_dim != REQUIRED_DIM:
                # DELETE: It's old (Dim 6) or incorrect
                os.remove(file_path)
                deleted_count += 1
            else:
                # KEEP: It's the good new data
                kept_count += 1
                
        except Exception as e:
            # Corrupt file? Delete it to be safe
            print(f"Error reading {file_path.name}: {e}. Deleting.")
            os.remove(file_path)
            error_count += 1

    print("\n=== Cleanup Complete ===")
    print(f"✅ Kept (New Physics): {kept_count}")
    print(f"🗑️  Deleted (Old/Stale): {deleted_count}")
    print(f"⚠️  Corrupt/Errors:      {error_count}")

if __name__ == "__main__":
    clean_directory()