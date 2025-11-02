# parse_results.py
import os
import json
import glob
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Directory where colabfold_batch saved everything
OUTPUT_DIR = Path(__file__).parent / "data/generated_pdbs"
FINAL_JSON = OUTPUT_DIR / "results.json"

def parse_scores():
    """
    Parses all _scores.json files to build a master results.json
    """
    logger.info(f"Looking for score files in: {OUTPUT_DIR}")
    
    # Find all the score files
    score_files = glob.glob(str(OUTPUT_DIR / "*_scores.json"))
    logger.info(f"Found {len(score_files)} score files to parse.")
    
    all_results = {"success": {}, "failed": [], "config": {}}
    
    for score_file_path in score_files:
        try:
            # 1. Get the header (e.g., "0_01_01_9")
            # Filename is like: /path/to/0_01_01_9_scores.json
            base_name = os.path.basename(score_file_path)
            header = base_name.replace("_scores.json", "")
            
            # 2. Read the scores from that file
            with open(score_file_path, 'r') as f:
                scores = json.load(f)
            
            # 3. Find the matching PDB file
            # PDB file is like: /path/to/0_01_01_9_unrelaxed_rank_001...pdb
            pdb_file_pattern = str(OUTPUT_DIR / f"{header}_unrelaxed_rank_*.pdb")
            pdb_files = glob.glob(pdb_file_pattern)
            
            if not pdb_files:
                logger.warning(f"Could not find PDB for header: {header}")
                all_results["failed"].append(header)
                continue

            # We just need the filename, not the full path
            pdb_filename = os.path.basename(pdb_files[0])
            
            # 4. Store the data in your desired format
            all_results["success"][header] = {
                "pdb_file": pdb_filename,
                "mean_plddt": scores.get("plddt"), # Get average pLDDT
                "ptm": scores.get("ptm"), # Get pTM score
                "full_scores": scores # Store the rest just in case
            }

        except Exception as e:
            logger.error(f"Failed to parse {score_file_path}: {e}")
            if 'header' in locals():
                all_results["failed"].append(header)

    # 5. Save the final, single results.json
    with open(FINAL_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    logger.info("="*70)
    logger.info(f"✓ Success: {len(all_results['success'])}/{len(score_files)}")
    logger.info(f"✗ Failed:  {len(all_results['failed'])}/{len(score_files)}")
    logger.info(f"✓ Master results file saved to: {FINAL_JSON}")

if __name__ == "__main__":
    parse_scores()