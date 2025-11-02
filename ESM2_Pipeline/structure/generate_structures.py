"""
Generate real PDB structures using ESM2-based structure prediction
Direct import - fixes fair_esm import issue
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import torch
from transformers import AutoTokenizer, AutoModel


class DirectESMFoldGenerator:
    """Generate structures using ESM2 + local folding"""
    
    def __init__(self, output_dir: str = "./data/generated_pdbs"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ESM2 Structure Generator initialized")
        logger.info(f"Output: {output_dir}")
        
        # Try direct import of esmfold
        try:
            from esmfold.esmfold import ESMFold
            self.ESMFold = ESMFold
            logger.info("✓ ESMFold imported successfully")
        except ImportError as e:
            logger.error(f"ESMFold import failed: {e}")
            logger.info("Trying alternative import method...")
            try:
                import fair_esm.models.esmfold as esmfold_module
                self.ESMFold = esmfold_module.ESMFold
                logger.info("✓ ESMFold imported via fair_esm")
            except ImportError as e2:
                logger.error(f"All ESMFold imports failed: {e2}")
                raise RuntimeError("Cannot import ESMFold. Try: pip install 'fair-esm[esmfold]' --force-reinstall")
    
    def predict(
        self,
        mhc_seq: str,
        pep_seq: str,
        mhc_id: str,
        pep_id: str,
        index: int
    ) -> Tuple[bool, str]:
        """Generate structure"""
        
        pdb_file = os.path.join(
            self.output_dir,
            f"complex_{index:06d}_{mhc_id}_{pep_id}.pdb"
        )
        
        if os.path.exists(pdb_file):
            logger.info(f"✓ Already exists: {Path(pdb_file).name}")
            return True, pdb_file
        
        logger.info(f"[{index}] Predicting: {mhc_id}({len(mhc_seq)}aa):{pep_id}({len(pep_seq)}aa)")
        
        try:
            # Combine sequences
            sequence = f"{mhc_seq}:{pep_seq}"
            
            # Load ESMFold model
            model = self.ESMFold.esmfold_structure_module("esmfold_structure_v0", "cpu")
            
            with torch.no_grad():
                # Get structure
                pdb_lines = model.infer_pdb(sequence)
            
            # Save
            with open(pdb_file, 'w') as f:
                f.write(pdb_lines)
            
            logger.info(f"✓ Saved: {Path(pdb_file).name}")
            return True, pdb_file
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return False, ""
    
    def generate_batch(
        self,
        data_csv: str,
        max_samples: int = None
    ) -> Dict[int, str]:
        """Generate batch"""
        
        logger.info(f"Loading {data_csv}")
        df = pd.read_csv(data_csv, on_bad_lines='skip')
        
        if max_samples:
            df = df.head(max_samples)
        
        logger.info(f"Generating {len(df)} structures")
        
        pdb_files = {}
        failed = []
        
        for idx, row in df.iterrows():
            try:
                mhc_seq = str(row['mhc_sequence']).strip()
                pep_seq = str(row['peptide']).strip()
                
                if not (mhc_seq and pep_seq):
                    failed.append(idx)
                    continue
                
                mhc_id = str(row.get('mhc_allele', 'MHC')).split('|')[0][:10]
                
                success, pdb_file = self.predict(
                    mhc_seq, pep_seq, mhc_id, "PEP", idx
                )
                
                if success:
                    pdb_files[idx] = pdb_file
                else:
                    failed.append(idx)
            except Exception as e:
                logger.warning(f"Sample {idx}: {e}")
                failed.append(idx)
        
        # Save mapping
        mapping = {
            "success": {str(k): v for k, v in pdb_files.items()},
            "failed": failed,
            "total": len(df),
            "backend": "esmfold"
        }
        
        with open(os.path.join(self.output_dir, "mapping.json"), 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"✓ Generated {len(pdb_files)}/{len(df)} structures")
        return pdb_files


if __name__ == "__main__":
    try:
        gen = DirectESMFoldGenerator()
        pdb_files = gen.generate_batch("../iedb_dataset.csv", max_samples=10)
        
        if pdb_files:
            print(f"\n✓ Generated {len(pdb_files)} real structures!")
        else:
            print("\n✗ No structures generated")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTry fixing with:")
        print("  pip install 'fair-esm[esmfold]' --force-reinstall --no-cache")