import os
import requests
import warnings

# Use standard RCSB HTTP link (firewall friendly)
BASE_URL = "https://files.rcsb.org/download/"
TEMPLATES = ["1HHI", "1A1M", "3VCL", "1XR8"]
DIR = "templates"

os.makedirs(DIR, exist_ok=True)

def download_pdb(pdb_id, dest_dir):
    url = f"{BASE_URL}{pdb_id.upper()}.pdb"
    dest_path = os.path.join(dest_dir, f"template_{pdb_id.lower()}.pdb")
    
    print(f"⬇️ Downloading {pdb_id} from {url}...")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Saved: {dest_path}")
        else:
            print(f"❌ Failed to download {pdb_id} (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    for t in TEMPLATES:
        download_pdb(t, DIR)