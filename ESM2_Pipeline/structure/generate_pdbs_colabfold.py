#!/usr/bin/env python
"""
ColabFold High-VRAM Optimization
For A100 with 30GB available - aggressive batch processing
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import tempfile
import logging
import json
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ColabFoldHighVRAM:
    """Optimized for maximum throughput with available VRAM"""
    
    def __init__(self, output_dir: str = "./data/generated_pdbs", batch_size: int = 200, num_recycle: int = 3):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_recycle = num_recycle  # Adjustable: 1=fastest, 3=best quality
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 70)
        logger.info(f"COLABFOLD HIGH-VRAM OPTIMIZATION")
        logger.info(f"Batch size: {batch_size} | Recycles: {num_recycle}")
        logger.info("=" * 70)
        
        self._check_environment()
    
    def _check_environment(self):
        """Check GPU and VRAM"""
        logger.info("[SETUP] Checking environment...")
        
        try:
            from colabfold.batch import main as batch_main
            logger.info(f"  ✓ ColabFold batch API available")
        except ImportError as e:
            logger.error(f"  ✗ ColabFold batch not found: {e}")
            raise
        
        try:
            import jax
            from jax import devices
            import jax.numpy as jnp
            
            gpu_devices = devices('gpu')
            if gpu_devices:
                logger.info(f"  ✓ GPU available: {len(gpu_devices)} device(s)")
                
                # Try to get GPU memory info
                try:
                    for i, device in enumerate(gpu_devices):
                        logger.info(f"    GPU {i}: {device}")
                except:
                    pass
            else:
                logger.warning("  ⚠ No GPU detected")
        except:
            logger.warning("  ⚠ Could not detect GPU")
    
    def process_batch(self, sequences: List[Dict], batch_idx: int) -> Dict:
        """Process batch with optimized recycle settings"""
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"BATCH {batch_idx}: {len(sequences)} sequences | {self.num_recycle} recycles")
        logger.info(f"{'=' * 70}")
        
        try:
            from colabfold.batch import get_queries, run
            
            # Create FASTA with all sequences
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
                for seq_data in sequences:
                    mhc_seq = seq_data['mhc_seq'].upper()
                    pep_seq = seq_data['pep_seq'].upper()
                    combined_seq = f"{mhc_seq}GGGGS{pep_seq}"
                    
                    header = f"{seq_data['idx']}_{seq_data['mhc_id']}_{seq_data['pep_id']}"
                    f.write(f">{header}\n{combined_seq}\n")
                
                fasta_file = f.name
            
            try:
                logger.info(f"[BATCH] Running ColabFold inference...")
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    queries, is_complex = get_queries(fasta_file)
                    logger.info(f"  ✓ Generated MSAs for {len(queries)} sequences")
                    
                    # Run with configurable recycles
                    results = run(
                        queries=queries,
                        result_dir=tmpdir,
                        num_models=1,
                        is_complex=is_complex,
                        use_templates=False,
                        use_amber=False,
                        num_recycle=self.num_recycle,  # ← CONFIGURABLE
                        recycle_early_stop_threshold=1.0,
                    )
                    
                    batch_results = self._process_results(tmpdir, sequences)
                    return batch_results
            
            finally:
                os.unlink(fasta_file)
        
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": {}, "failed": [s['idx'] for s in sequences]}
    
    def _process_results(self, result_dir: str, sequences: List[Dict]) -> Dict:
        """Extract results"""
        
        success = {}
        failed = []
        seq_map = {f"{s['idx']}_{s['mhc_id']}_{s['pep_id']}": s for s in sequences}
        pdb_files = list(Path(result_dir).glob("*.pdb"))
        
        for pdb_file in pdb_files:
            try:
                stem = pdb_file.stem
                header = None
                
                for seq_header in seq_map.keys():
                    if seq_header in stem:
                        header = seq_header
                        break
                
                if not header:
                    continue
                
                seq_data = seq_map[header]
                output_file = os.path.join(
                    self.output_dir,
                    f"complex_{seq_data['idx']:06d}_{seq_data['mhc_id']}_{seq_data['pep_id']}.pdb"
                )
                
                with open(pdb_file, 'r') as src:
                    pdb_content = src.read()
                
                with open(output_file, 'w') as dst:
                    dst.write(pdb_content)
                
                mean_plddt = self._extract_plddt(pdb_content)
                success[seq_data['idx']] = {"pdb_file": output_file, "mean_plddt": mean_plddt}
                
                logger.info(f"  ✓ [{seq_data['idx']}] pLDDT={mean_plddt:.1f}")
            
            except Exception as e:
                logger.error(f"  Error: {e}")
                if 'seq_data' in locals():
                    failed.append(seq_data['idx'])
        
        return {"success": success, "failed": failed}
    
    def _extract_plddt(self, pdb_string: str) -> float:
        """Extract mean pLDDT"""
        try:
            plddt_values = [float(line[60:66].strip()) for line in pdb_string.split('\n') 
                          if line.startswith('ATOM')]
            return sum(plddt_values) / len(plddt_values) if plddt_values else 0.0
        except:
            return 0.0
    
    def process_csv(self, csv_path: str, mhc_col: str = "mhc_sequence", pep_col: str = "peptide",
                   mhc_id_col: str = "mhc_allele", max_samples: int = None, start_idx: int = 0) -> Dict:
        """Process CSV dataset"""
        
        logger.info("\n" + "=" * 70)
        logger.info("LOADING CSV")
        logger.info("=" * 70)
        
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        logger.info(f"Total rows: {len(rows)}")
        
        if max_samples:
            rows = rows[:max_samples]
            logger.info(f"Limited to: {len(rows)} samples")
        
        sequences = []
        for row_idx, row in enumerate(rows):
            actual_idx = start_idx + row_idx
            
            mhc_seq = str(row.get(mhc_col, "")).strip().upper()
            pep_seq = str(row.get(pep_col, "")).strip().upper()
            
            if not mhc_seq or not pep_seq:
                continue
            
            mhc_id = str(row.get(mhc_id_col, "MHC")).split("*")[-1][:10]
            pep_id = str(len(pep_seq))
            
            sequences.append({
                "idx": actual_idx,
                "mhc_seq": mhc_seq,
                "pep_seq": pep_seq,
                "mhc_id": mhc_id,
                "pep_id": pep_id
            })
        
        logger.info(f"✓ Prepared {len(sequences)} sequences")
        
        # Process in batches
        all_results = {"success": {}, "failed": []}
        
        for batch_idx in range(0, len(sequences), self.batch_size):
            batch = sequences[batch_idx:batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            
            batch_results = self.process_batch(batch, batch_num)
            all_results["success"].update(batch_results["success"])
            all_results["failed"].extend(batch_results["failed"])
            
            total_done = len(all_results["success"])
            logger.info(f"\n[OVERALL] {total_done}/{len(sequences)} completed")
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"✓ Success: {len(all_results['success'])}/{len(sequences)}")
        logger.info(f"✗ Failed: {len(all_results['failed'])}/{len(sequences)}")
        
        output_file = os.path.join(self.output_dir, "results.json")
        save_data = {
            "success": {
                str(k): {"pdb_file": v["pdb_file"], "mean_plddt": v.get("mean_plddt")}
                for k, v in all_results["success"].items()
            },
            "failed": all_results["failed"],
            "config": {"batch_size": self.batch_size, "num_recycle": self.num_recycle}
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"✓ Results saved: {output_file}")
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ColabFold High-VRAM Optimization")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument("--output", default="./data/generated_pdbs", help="Output dir")
    parser.add_argument("--batch-size", type=int, default=200, help="Sequences per batch")
    parser.add_argument("--num-recycle", type=int, default=3, help="Recycle iterations (1=fast, 3=best)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index")
    
    args = parser.parse_args()
    
    runner = ColabFoldHighVRAM(
        output_dir=args.output,
        batch_size=args.batch_size,
        num_recycle=args.num_recycle
    )
    results = runner.process_csv(
        csv_path=args.csv,
        max_samples=args.max_samples,
        start_idx=args.start_idx
    )
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())