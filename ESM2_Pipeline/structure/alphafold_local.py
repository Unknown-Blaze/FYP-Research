#!/usr/bin/env python
"""
Local AlphaFold Runner - Uses Local Parameters (No Internet Required)
Generates MHC-Peptide 3D structures using AlphaFold Multimer weights
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Add alphafold to path
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# The official alphafold code is in a subdirectory named 'alphafold'
ALPHAFOLD_DIR = os.path.join(SCRIPT_DIR, "alphafold")
sys.path.insert(0, ALPHAFOLD_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalAlphaFoldRunner:
    """
    Run AlphaFold locally using pre-downloaded parameters
    Optimized for MHC-Peptide complexes
    """
    
    def __init__(
        self,
        params_dir: str = "./params",
        output_dir: str = "./data/generated_pdbs",
        model_name: str = "model_1_multimer_v3",
        device: str = "cuda"
    ):
        self.params_dir = params_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.device = device
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("LOCAL ALPHAFOLD MULTIMER RUNNER")
        logger.info("="*60)
        logger.info(f"Params dir: {params_dir}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        
        self.model = None
        self.model_params = None
        self._load_model()
    
    def _load_model(self):
        """Load AlphaFold model and parameters"""
        try:
            import jax
            import haiku as hk
            from alphafold.model import model, config
            from alphafold.data import pipeline, templates
            
            logger.info("Loading AlphaFold model...")
            
            # Load model config (multimer for complexes)
            model_config = config.model_config(self.model_name)
            
            # Adjust config for faster prediction (optional)
            model_config.data.eval.num_ensemble = 1
            model_config.model.num_recycle = 3  # Default: 3 (more = better but slower)
            
            # Load model parameters
            params_path = os.path.join(self.params_dir, f"params_{self.model_name}.npz")
            
            if not os.path.exists(params_path):
                logger.error(f"Parameters not found: {params_path}")
                logger.error(f"Available models in {self.params_dir}:")
                for f in Path(self.params_dir).glob("params_model_*_multimer_v3.npz"):
                    logger.error(f"  - {f.name}")
                raise FileNotFoundError(f"Parameters not found: {params_path}")
            
            logger.info(f"Loading parameters from: {params_path}")
            with open(params_path, 'rb') as f:
                self.model_params = np.load(f, allow_pickle=False)
            
            # Initialize model
            self.model_runner = model.RunModel(model_config, self.model_params)
            
            logger.info("✓ AlphaFold model loaded successfully")
            logger.info(f"  Num recycles: {model_config.model.num_recycle}")
            logger.info(f"  Num ensemble: {model_config.data.eval.num_ensemble}")
            
        except ImportError as e:
            logger.error(f"AlphaFold dependencies missing: {e}")
            logger.error("\nRequired packages:")
            logger.error("  - jax")
            logger.error("  - haiku")
            logger.error("  - alphafold (from alphafold_finetune)")
            logger.error("\nInstall with:")
            logger.error("  pip install dm-haiku jax jaxlib")
            raise
        except Exception as e:
            logger.error(f"Error loading AlphaFold: {e}")
            raise
    
    def predict_structure(
        self,
        mhc_seq: str,
        pep_seq: str,
        index: int,
        mhc_id: str = "MHC",
        pep_id: str = "PEP"
    ) -> Tuple[bool, str]:
        """
        Predict 3D structure for MHC-peptide complex
        
        Args:
            mhc_seq: MHC sequence
            pep_seq: Peptide sequence
            index: Sample index
            mhc_id: MHC identifier
            pep_id: Peptide identifier
        
        Returns:
            (success, pdb_file_path)
        """
        pdb_file = os.path.join(
            self.output_dir,
            f"complex_{index:06d}_{mhc_id}_{pep_id}.pdb"
        )
        
        # Skip if exists
        if os.path.exists(pdb_file):
            logger.info(f"✓ [{index}] Structure exists: {os.path.basename(pdb_file)}")
            return True, pdb_file
        
        logger.info(f"[{index}] Predicting structure: {mhc_id}:{pep_id}")
        logger.info(f"  MHC length: {len(mhc_seq)}")
        logger.info(f"  Peptide length: {len(pep_seq)}")
        
        try:
            from alphafold.common import protein, residue_constants
            from alphafold.data import pipeline_multimer
            import jax.numpy as jnp
            
            # Prepare input features for multimer
            sequences = [mhc_seq, pep_seq]
            descriptions = [mhc_id, pep_id]
            
            # Create feature dict (simplified - no MSA or templates for speed)
            feature_dict = self._make_features(sequences, descriptions)
            
            # Run prediction
            logger.info(f"  Running AlphaFold prediction...")
            prediction_result = self.model_runner.predict(feature_dict, random_seed=index)
            
            # Extract structure
            plddt = prediction_result['plddt']
            unrelaxed_protein = protein.from_prediction(
                features=feature_dict,
                result=prediction_result,
                b_factors=plddt
            )
            
            # Save PDB
            pdb_string = protein.to_pdb(unrelaxed_protein)
            with open(pdb_file, 'w') as f:
                f.write(pdb_string)
            
            # Log confidence scores
            mean_plddt = np.mean(plddt)
            logger.info(f"✓ [{index}] Structure saved: {os.path.basename(pdb_file)}")
            logger.info(f"  Mean pLDDT: {mean_plddt:.1f} (confidence score)")
            
            return True, pdb_file
            
        except Exception as e:
            logger.error(f"✗ [{index}] Prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, ""
    
    def _make_features(self, sequences, descriptions):
        """
        Create minimal feature dict for AlphaFold Multimer
        This is a simplified version without MSA search (faster but lower accuracy)
        """
        from alphafold.data import pipeline_multimer
        from alphafold.common import residue_constants
        import jax.numpy as jnp
        
        num_res = sum(len(seq) for seq in sequences)
        
        # Create sequence features
        feature_dict = {
            'aatype': np.zeros(num_res, dtype=np.int32),
            'residue_index': np.arange(num_res, dtype=np.int32),
            'seq_length': np.array([num_res], dtype=np.int32),
            'num_alignments': np.array([1], dtype=np.int32),
            'seq_mask': np.ones(num_res, dtype=np.float32),
        }
        
        # Convert sequences to aatype
        offset = 0
        for seq in sequences:
            for i, aa in enumerate(seq):
                if aa in residue_constants.restype_order:
                    feature_dict['aatype'][offset + i] = residue_constants.restype_order[aa]
                else:
                    # Unknown residue -> use 'X' (index 20)
                    feature_dict['aatype'][offset + i] = 20
            offset += len(seq)
        
        # Create MSA features (single sequence - no MSA search for speed)
        feature_dict['msa'] = feature_dict['aatype'][None]  # Shape: (1, num_res)
        feature_dict['deletion_matrix'] = np.zeros((1, num_res), dtype=np.float32)
        
        # Multimer features
        feature_dict['asym_id'] = np.zeros(num_res, dtype=np.int32)
        feature_dict['entity_id'] = np.zeros(num_res, dtype=np.int32)
        feature_dict['sym_id'] = np.zeros(num_res, dtype=np.int32)
        
        # Assign chain IDs
        offset = 0
        for chain_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            feature_dict['asym_id'][offset:offset+seq_len] = chain_idx
            feature_dict['entity_id'][offset:offset+seq_len] = chain_idx
            feature_dict['sym_id'][offset:offset+seq_len] = 0
            offset += seq_len
        
        # Template features (empty - no templates)
        feature_dict['template_aatype'] = np.zeros((0, num_res), dtype=np.int32)
        feature_dict['template_all_atom_masks'] = np.zeros((0, num_res, 37), dtype=np.float32)
        feature_dict['template_all_atom_positions'] = np.zeros((0, num_res, 37, 3), dtype=np.float32)
        
        return feature_dict
    
    def generate_batch(
        self,
        data_csv: str,
        mhc_col: str = "mhc_sequence",
        pep_col: str = "peptide",
        mhc_id_col: str = "mhc_allele",
        pep_id_col: str = "peptide_id",
        max_samples: Optional[int] = None,
        start_idx: int = 0,
        max_length: int = 1500
    ) -> Dict[int, str]:
        """
        Generate structures for all samples in dataset
        
        Args:
            data_csv: Path to CSV file
            max_samples: Limit number of samples
            max_length: Skip sequences longer than this (GPU memory limit)
        
        Returns:
            Dictionary mapping indices to PDB paths
        """
        logger.info("="*60)
        logger.info(f"Loading dataset: {data_csv}")
        
        df = pd.read_csv(data_csv, on_bad_lines='skip')
        
        if max_samples:
            df = df.head(max_samples)
        
        logger.info(f"Total samples: {len(df)}")
        logger.info("="*60)
        
        pdb_files = {}
        failed = []
        skipped_long = []
        
        for idx, row in df.iterrows():
            actual_idx = start_idx + idx
            
            try:
                mhc_seq = str(row[mhc_col]).strip()
                pep_seq = str(row[pep_col]).strip()
                
                # Get IDs
                mhc_id = str(row.get(mhc_id_col, "MHC")).replace('/', '_').replace('*', '')[:20]
                pep_id = str(row.get(pep_id_col, f"PEP{idx}"))[:20]
                
                # Validate
                if not self._is_valid_sequence(mhc_seq):
                    logger.warning(f"[{actual_idx}] Invalid MHC sequence")
                    failed.append(actual_idx)
                    continue
                
                if not self._is_valid_sequence(pep_seq):
                    logger.warning(f"[{actual_idx}] Invalid peptide sequence")
                    failed.append(actual_idx)
                    continue
                
                # Check length
                total_len = len(mhc_seq) + len(pep_seq)
                if total_len > max_length:
                    logger.warning(f"[{actual_idx}] Too long ({total_len} > {max_length}), skipping")
                    skipped_long.append(actual_idx)
                    continue
                
                # Predict
                success, pdb_file = self.predict_structure(
                    mhc_seq, pep_seq, actual_idx, mhc_id, pep_id
                )
                
                if success:
                    pdb_files[actual_idx] = pdb_file
                else:
                    failed.append(actual_idx)
                
                # Progress
                if (idx + 1) % 5 == 0:
                    logger.info(f"\nProgress: {idx + 1}/{len(df)} ({len(pdb_files)} successful)\n")
                    
            except Exception as e:
                logger.error(f"[{actual_idx}] Error: {e}")
                failed.append(actual_idx)
        
        # Save mapping
        self._save_mapping(pdb_files, failed, skipped_long)
        
        logger.info("\n" + "="*60)
        logger.info("GENERATION COMPLETE")
        logger.info("="*60)
        logger.info(f"✓ Success: {len(pdb_files)}/{len(df)}")
        logger.info(f"✗ Failed: {len(failed)}")
        logger.info(f"⊘ Skipped (too long): {len(skipped_long)}")
        logger.info("="*60)
        
        return pdb_files
    
    def _is_valid_sequence(self, seq: str) -> bool:
        """Validate protein sequence"""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        return len(seq) > 0 and all(aa.upper() in valid_aas for aa in seq)
    
    def _save_mapping(self, pdb_files: Dict[int, str], failed: list, skipped: list):
        """Save mapping file"""
        mapping = {
            "success": {str(k): v for k, v in pdb_files.items()},
            "failed": failed,
            "skipped_long": skipped,
            "total": len(pdb_files) + len(failed) + len(skipped),
            "backend": "local_alphafold_multimer",
            "model": self.model_name
        }
        
        mapping_file = os.path.join(self.output_dir, "mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Mapping saved: {mapping_file}")


def main():
    """Run structure generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local AlphaFold structure generation")
    parser.add_argument("--csv", default="../iedb_dataset.csv", help="Input CSV")
    parser.add_argument("--output", default="./data/generated_pdbs", help="Output directory")
    parser.add_argument("--params", default="./params", help="AlphaFold parameters directory")
    parser.add_argument("--model", default="model_1_multimer_v3", help="Model name")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for testing)")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--max-length", type=int, default=1500, help="Max total sequence length")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LOCAL ALPHAFOLD STRUCTURE GENERATION")
    print("="*60)
    print(f"CSV: {args.csv}")
    print(f"Output: {args.output}")
    print(f"Parameters: {args.params}")
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Start index: {args.start_idx}")
    print("="*60 + "\n")
    
    # Initialize
    runner = LocalAlphaFoldRunner(
        params_dir=args.params,
        output_dir=args.output,
        model_name=args.model
    )
    
    # Generate structures
    pdb_files = runner.generate_batch(
        data_csv=args.csv,
        max_samples=args.max_samples,
        start_idx=args.start_idx,
        max_length=args.max_length
    )
    
    if pdb_files:
        print(f"\n✓ Generated {len(pdb_files)} structures")
        print(f"✓ Output: {args.output}")
        print(f"\nNext: Build graphs")
        print(f"  python build_graphs.py")
        return 0
    else:
        print("\n✗ No structures generated")
        return 1


if __name__ == "__main__":
    sys.exit(main())
