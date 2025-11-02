#!/usr/bin/env python
"""
Local ESMFold Structure Generation (No Internet Required)
Uses Meta's ESM models for fast local structure prediction
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalESMFoldGenerator:
    """Generate structures using ESM models locally (no API calls)"""
    
    def __init__(self, output_dir: str = "./data/generated_pdbs", device: str = "cuda"):
        self.output_dir = output_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Local ESMFold Generator")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output: {output_dir}")
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load ESMFold model"""
        try:
            import esm
            logger.info("Loading ESMFold model (this may take a few minutes)...")
            
            # Load ESMFold v1 model
            self.model = esm.pretrained.esmfold_v1()
            self.model = self.model.eval().to(self.device)
            
            logger.info("✓ ESMFold model loaded successfully")
            
        except ImportError:
            logger.error("ESM not installed. Install with:")
            logger.error("  pip install fair-esm")
            logger.error("\nOr use alternative:")
            logger.error("  pip install git+https://github.com/facebookresearch/esm.git")
            raise
        except Exception as e:
            logger.error(f"Error loading ESMFold: {e}")
            logger.info("\nAlternative: You can also try:")
            logger.info("1. Use pre-generated structures from AlphaFold DB")
            logger.info("2. Generate locally with localcolabfold")
            logger.info("3. Use a simpler homology modeling approach")
            raise
    
    def predict_structure(
        self,
        mhc_seq: str,
        pep_seq: str,
        index: int,
        mhc_id: str = "MHC",
        pep_id: str = "PEP"
    ) -> tuple[bool, str]:
        """
        Generate structure for MHC-peptide complex
        
        Args:
            mhc_seq: MHC sequence
            pep_seq: Peptide sequence
            index: Sample index
            mhc_id: MHC identifier (for filename)
            pep_id: Peptide identifier (for filename)
        
        Returns:
            (success, pdb_file_path)
        """
        pdb_file = os.path.join(
            self.output_dir,
            f"complex_{index:06d}_{mhc_id}_{pep_id}.pdb"
        )
        
        # Skip if exists
        if os.path.exists(pdb_file):
            logger.info(f"✓ [{index}] Structure already exists: {pdb_file}")
            return True, pdb_file
        
        logger.info(f"[{index}] Generating structure for {mhc_id}:{pep_id}")
        
        try:
            # Combine sequences (ESMFold handles complexes as concatenated sequences)
            full_sequence = f"{mhc_seq}:{pep_seq}"
            
            # Predict structure
            with torch.no_grad():
                output = self.model.infer_pdb(full_sequence)
            
            # Save PDB
            with open(pdb_file, 'w') as f:
                f.write(output)
            
            logger.info(f"✓ [{index}] Structure saved: {pdb_file}")
            return True, pdb_file
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"[{index}] GPU out of memory. Try:")
                logger.error("  - Reduce batch size")
                logger.error("  - Use CPU (slower): device='cpu'")
                logger.error("  - Skip very long sequences")
            else:
                logger.error(f"[{index}] Runtime error: {e}")
            return False, ""
            
        except Exception as e:
            logger.error(f"[{index}] Prediction failed: {e}")
            return False, ""
    
    def generate_batch(
        self,
        data_csv: str,
        mhc_col: str = "mhc_sequence",
        pep_col: str = "peptide",
        mhc_id_col: str = "mhc_allele",
        pep_id_col: str = "peptide_id",
        max_samples: Optional[int] = None,
        max_length: int = 1000
    ) -> Dict[int, str]:
        """
        Generate structures for all samples in CSV
        
        Args:
            data_csv: Path to dataset CSV
            mhc_col: MHC sequence column
            pep_col: Peptide sequence column
            max_samples: Limit number of samples (for testing)
            max_length: Skip sequences longer than this
        
        Returns:
            Dictionary mapping indices to PDB file paths
        """
        logger.info(f"Loading data from {data_csv}")
        df = pd.read_csv(data_csv, on_bad_lines='skip')
        
        if max_samples:
            df = df.head(max_samples)
        
        logger.info(f"Generating structures for {len(df)} complexes")
        
        pdb_files = {}
        failed = []
        skipped_long = []
        
        for idx, row in df.iterrows():
            try:
                mhc_seq = str(row[mhc_col]).strip()
                pep_seq = str(row[pep_col]).strip()
                
                # Get IDs if available
                mhc_id = str(row.get(mhc_id_col, "MHC")).replace('/', '_')[:20]
                pep_id = str(row.get(pep_id_col, "PEP")).replace('/', '_')[:20]
                
                # Validate sequences
                if not self._is_valid_sequence(mhc_seq) or not self._is_valid_sequence(pep_seq):
                    logger.warning(f"[{idx}] Invalid sequence, skipping")
                    failed.append(idx)
                    continue
                
                # Skip very long sequences (ESMFold has limits)
                total_length = len(mhc_seq) + len(pep_seq)
                if total_length > max_length:
                    logger.warning(f"[{idx}] Sequence too long ({total_length} > {max_length}), skipping")
                    skipped_long.append(idx)
                    continue
                
                # Generate structure
                success, pdb_file = self.predict_structure(
                    mhc_seq, pep_seq, idx, mhc_id, pep_id
                )
                
                if success:
                    pdb_files[idx] = pdb_file
                else:
                    failed.append(idx)
                
                # Progress update
                if (idx + 1) % 10 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(df)} ({len(pdb_files)} successful)")
                    
            except Exception as e:
                logger.error(f"[{idx}] Error: {e}")
                failed.append(idx)
        
        # Save mapping
        self._save_mapping(pdb_files, failed, skipped_long)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Successfully generated {len(pdb_files)}/{len(df)} structures")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Skipped (too long): {len(skipped_long)}")
        logger.info(f"{'='*60}")
        
        return pdb_files
    
    def _is_valid_sequence(self, seq: str) -> bool:
        """Check if sequence contains valid amino acids"""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        return len(seq) > 0 and all(aa.upper() in valid_aas for aa in seq)
    
    def _save_mapping(self, pdb_files: Dict[int, str], failed: list, skipped: list):
        """Save mapping file"""
        mapping = {
            "success": {str(k): v for k, v in pdb_files.items()},
            "failed": failed,
            "skipped_long": skipped,
            "total": len(pdb_files) + len(failed) + len(skipped),
            "backend": "local_esmfold"
        }
        
        mapping_file = os.path.join(self.output_dir, "mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Mapping saved to: {mapping_file}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local ESMFold structure generation")
    parser.add_argument("--csv", default="../iedb_dataset.csv", help="Input CSV file")
    parser.add_argument("--output", default="./data/generated_pdbs", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=10, help="Max samples (for testing)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--max-length", type=int, default=800, help="Max sequence length")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LOCAL ESMFOLD STRUCTURE GENERATION")
    print("="*60)
    print(f"Input: {args.csv}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples}")
    print("="*60 + "\n")
    
    # Initialize generator
    generator = LocalESMFoldGenerator(
        output_dir=args.output,
        device=args.device
    )
    
    # Generate structures
    pdb_files = generator.generate_batch(
        data_csv=args.csv,
        max_samples=args.max_samples,
        max_length=args.max_length
    )
    
    if pdb_files:
        print(f"\n✓ Success! Generated {len(pdb_files)} PDB files")
        print(f"✓ Output: {args.output}")
        print(f"\nNext step: Build graphs")
        print(f"  python build_graphs.py")
    else:
        print("\n✗ No structures generated. Check errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
