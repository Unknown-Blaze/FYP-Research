#!/usr/bin/env python
"""
Generate mapping.json from existing PDB files
Use this if you already ran structure generation and want to add traceability
"""

import os
import csv
import json
from pathlib import Path


def create_mapping_from_csv(csv_path, pdb_dir, output_json):
    """
    Create mapping.json by reading CSV and matching to PDB files
    
    Args:
        csv_path: Path to iedb_dataset.csv
        pdb_dir: Directory containing PDB files
        output_json: Where to save mapping.json
    """
    
    print("Reading CSV...")
    
    # Load CSV
    sequences = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            mhc_seq = str(row.get('mhc_sequence', '')).strip().upper()
            pep_seq = str(row.get('peptide', '')).strip().upper()
            mhc_allele = str(row.get('mhc_allele', '')).strip()
            
            if not mhc_seq or not pep_seq:
                continue
            
            # Extract allele code (last part after *)
            mhc_id = mhc_allele.split('*')[-1][:10] if mhc_allele else 'MHC'
            pep_id = len(pep_seq)
            
            sequences[row_idx] = {
                'mhc_seq': mhc_seq,
                'pep_seq': pep_seq,
                'mhc_allele': mhc_allele,
                'mhc_id': mhc_id,
                'pep_id': pep_id,
                'combined_seq': f"{mhc_seq}GGGGS{pep_seq}"
            }
    
    print(f"Loaded {len(sequences)} sequences from CSV")
    
    # Read existing PDB files
    print("\nReading existing PDB files...")
    pdb_files = list(Path(pdb_dir).glob('complex_*.pdb'))
    print(f"Found {len(pdb_files)} PDB files")
    
    mapping = {}
    
    for pdb_file in pdb_files:
        filename = pdb_file.name
        # Extract index from filename: complex_000000_01:01_10.pdb
        parts = filename.replace('complex_', '').replace('.pdb', '').split('_')
        
        try:
            idx = int(parts[0])
            
            if idx not in sequences:
                print(f"Warning: Index {idx} from {filename} not in CSV")
                continue
            
            # Extract pLDDT from PDB b-factor
            plddt_values = []
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM'):
                        try:
                            bfactor = float(line[60:66].strip())
                            plddt_values.append(bfactor)
                        except:
                            pass
            
            mean_plddt = sum(plddt_values) / len(plddt_values) if plddt_values else None
            
            seq_data = sequences[idx]
            mapping[str(idx)] = {
                'pdb_file': filename,
                'csv_row_index': idx,
                'mhc_sequence': seq_data['mhc_seq'],
                'peptide_sequence': seq_data['pep_seq'],
                'mhc_allele': seq_data['mhc_allele'],
                'mhc_id': seq_data['mhc_id'],
                'peptide_length': seq_data['pep_id'],
                'mhc_length': len(seq_data['mhc_seq']),
                'combined_sequence': seq_data['combined_seq'],
                'mean_plddt': mean_plddt
            }
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(pdb_files)} PDB files")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Save mapping
    print(f"\nSaving {len(mapping)} entries to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✓ Mapping saved: {output_json}")
    
    # Print statistics
    plddt_scores = [v['mean_plddt'] for v in mapping.values() if v['mean_plddt'] is not None]
    if plddt_scores:
        print(f"\nStatistics:")
        print(f"  Total structures: {len(mapping)}")
        print(f"  Average pLDDT: {sum(plddt_scores) / len(plddt_scores):.2f}")
        print(f"  Min pLDDT: {min(plddt_scores):.2f}")
        print(f"  Max pLDDT: {max(plddt_scores):.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mapping.json from existing PDB files")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--pdb-dir", required=True, help="Directory with PDB files")
    parser.add_argument("--output", default="mapping.json", help="Output mapping file")
    
    args = parser.parse_args()
    
    create_mapping_from_csv(args.csv, args.pdb_dir, args.output)