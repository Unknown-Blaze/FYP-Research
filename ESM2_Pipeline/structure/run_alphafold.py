"""
Use local AlphaFold installation to generate PDB structures
Works when API access is blocked but local installation exists
"""
import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalAlphaFoldRunner:
    """Generate structures using installed AlphaFold"""
    
    def __init__(
        self,
        output_dir: str = "./data/generated_pdbs",
        use_gpu: bool = True,
        num_recycles: int = 3
    ):
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.num_recycles = num_recycles
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Local AlphaFold Runner initialized")
        logger.info(f"GPU: {use_gpu}")
        logger.info(f"Recycles: {num_recycles}")
    
    def predict(
        self,
        mhc_seq: str,
        pep_seq: str,
        mhc_id: str,
        pep_id: str,
        index: int
    ) -> Tuple[bool, str]:
        """
        Generate structure using local AlphaFold
        
        Returns:
            (success, pdb_file_path)
        """
        pdb_file = os.path.join(
            self.output_dir,
            f"complex_{index:06d}_{mhc_id}_{pep_id}.pdb"
        )
        
        # Skip if already exists
        if os.path.exists(pdb_file):
            logger.info(f"✓ Already exists: {Path(pdb_file).name}")
            return True, pdb_file
        
        logger.info(f"[{index}] Generating: {mhc_id}({len(mhc_seq)}aa):{pep_id}({len(pep_seq)}aa)")
        
        # Create temporary FASTA
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(f">{mhc_id}\n{mhc_seq}\n")
            f.write(f">{pep_id}\n{pep_seq}\n")
            fasta_file = f.name
        
        try:
            # Create job output directory
            job_output_dir = os.path.join(self.output_dir, f"job_{index:06d}")
            os.makedirs(job_output_dir, exist_ok=True)
            
            # Build AlphaFold command
            cmd = [
                "python",
                "-m",
                "alphafold.run_alphafold",
                f"--fasta_paths={fasta_file}",
                f"--output_dir={job_output_dir}",
                "--db_preset=full_dbs",
                "--model_preset=multimer",  # Critical for MHC-peptide complex
                f"--num_recycles={self.num_recycles}",
                "--use_gpu_relax=True" if self.use_gpu else "--use_gpu_relax=False",
            ]
            
            logger.info(f"Running AlphaFold multimer...")
            
            # Run AlphaFold
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Find best PDB (ranked_0.pdb is usually best)
                pdb_candidates = [
                    os.path.join(job_output_dir, f) 
                    for f in os.listdir(job_output_dir) 
                    if f.endswith('.pdb')
                ]
                
                if not pdb_candidates:
                    # Try subdirectories
                    for root, dirs, files in os.walk(job_output_dir):
                        for f in files:
                            if f.endswith('.pdb'):
                                pdb_candidates.append(os.path.join(root, f))
                
                if pdb_candidates:
                    # Prefer ranked_0.pdb
                    best_pdb = None
                    for pdb in pdb_candidates:
                        if "ranked_0" in pdb:
                            best_pdb = pdb
                            break
                    if not best_pdb:
                        best_pdb = pdb_candidates[0]
                    
                    import shutil
                    shutil.copy(best_pdb, pdb_file)
                    
                    logger.info(f"✓ Structure saved: {Path(pdb_file).name}")
                    return True, pdb_file
                else:
                    logger.error(f"No PDB found in {job_output_dir}")
                    return False, ""
            else:
                error_msg = result.stderr.decode() if result.stderr else "Unknown error"
                logger.error(f"AlphaFold error: {error_msg[:200]}")
                return False, ""
        
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout after 1 hour for {index}")
            return False, ""
        except Exception as e:
            logger.error(f"Error: {e}")
            return False, ""
        finally:
            if os.path.exists(fasta_file):
                os.remove(fasta_file)
    
    def generate_batch(
        self,
        data_csv: str,
        max_samples: int = None
    ) -> Dict[int, str]:
        """Generate structures for all samples"""
        
        logger.info(f"Loading {data_csv}")
        df = pd.read_csv(data_csv, on_bad_lines='skip')
        
        if max_samples:
            df = df.head(max_samples)
        
        logger.info(f"Generating {len(df)} structures using local AlphaFold")
        logger.info("=" * 60)
        
        pdb_files = {}
        failed = []
        
        for idx, row in df.iterrows():
            try:
                mhc_seq = str(row['mhc_sequence']).strip()
                pep_seq = str(row['peptide']).strip()
                
                # Validate
                if not (mhc_seq and pep_seq):
                    logger.warning(f"Empty sequence at {idx}")
                    failed.append(idx)
                    continue
                
                mhc_id = str(row.get('mhc_allele', 'MHC')).split('|')[0][:10]
                pep_id = f"PEP_{idx}"
                
                success, pdb_file = self.predict(
                    mhc_seq, pep_seq, mhc_id, pep_id, idx
                )
                
                if success:
                    pdb_files[idx] = pdb_file
                else:
                    failed.append(idx)
            
            except Exception as e:
                logger.warning(f"Error at {idx}: {e}")
                failed.append(idx)
        
        # Save mapping
        mapping = {
            "success": {str(k): v for k, v in pdb_files.items()},
            "failed": failed,
            "total": len(df),
            "backend": "local_alphafold_multimer"
        }
        
        mapping_file = os.path.join(self.output_dir, "mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info("=" * 60)
        logger.info(f"✓ Successfully generated {len(pdb_files)}/{len(df)} structures")
        logger.info(f"✓ Output: {self.output_dir}")
        logger.info(f"✓ Mapping: {mapping_file}")
        
        return pdb_files


def main():
    """Generate structures using local AlphaFold"""
    
    runner = LocalAlphaFoldRunner(
        output_dir="./data/generated_pdbs",
        use_gpu=True,
        num_recycles=3  # Increase for better quality
    )
    
    pdb_files = runner.generate_batch(
        data_csv="../iedb_dataset.csv",
        max_samples=10  # Start small for testing
    )
    
    print("\n" + "=" * 60)
    if pdb_files:
        print(f"✓ Generated {len(pdb_files)} real AlphaFold structures!")
        print("✓ Next: Run 2_build_graphs.py to convert to graphs")
    else:
        print("✗ No structures generated")
    print("=" * 60)


if __name__ == "__main__":
    main()