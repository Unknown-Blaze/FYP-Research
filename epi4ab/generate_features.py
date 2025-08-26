# generate_features.py (Final Version with All Sequence Extraction)
import os
import pandas as pd
import numpy as np
import json
import subprocess
import tempfile
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.Polypeptide import is_aa
from Bio.Data import IUPACData
import freesasa
from scipy.spatial import KDTree
from tqdm import tqdm
import warnings
import argparse

# --- FIX for DSSP Configuration ---
cif_data_dir = os.path.expanduser('~/libcifpp_data')
if os.path.isdir(cif_data_dir):
    os.environ['LIBCIFPP_DATA_DIR'] = cif_data_dir
# ------------------------------------

# Suppress PDBConstructionWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

# --- Amino Acid Properties ---
AA_PROPERTIES = {
    'ALA': {'charge': 0, 'hydrophobicity': 1.8, 'volume': 88.6, 'polarity': 'nonpolar'}, 'CYS': {'charge': 0, 'hydrophobicity': 2.5, 'volume': 108.5, 'polarity': 'polar'},
    'ASP': {'charge': -1, 'hydrophobicity': -3.5, 'volume': 111.1, 'polarity': 'acidic'}, 'GLU': {'charge': -1, 'hydrophobicity': -3.5, 'volume': 138.4, 'polarity': 'acidic'},
    'PHE': {'charge': 0, 'hydrophobicity': 2.8, 'volume': 189.9, 'polarity': 'nonpolar'}, 'GLY': {'charge': 0, 'hydrophobicity': -0.4, 'volume': 60.1, 'polarity': 'nonpolar'},
    'HIS': {'charge': 1, 'hydrophobicity': -3.2, 'volume': 153.2, 'polarity': 'basic'}, 'ILE': {'charge': 0, 'hydrophobicity': 4.5, 'volume': 166.7, 'polarity': 'nonpolar'},
    'LYS': {'charge': 1, 'hydrophobicity': -3.9, 'volume': 168.6, 'polarity': 'basic'}, 'LEU': {'charge': 0, 'hydrophobicity': 3.8, 'volume': 166.7, 'polarity': 'nonpolar'},
    'MET': {'charge': 0, 'hydrophobicity': 1.9, 'volume': 162.9, 'polarity': 'nonpolar'}, 'ASN': {'charge': 0, 'hydrophobicity': -3.5, 'volume': 114.1, 'polarity': 'polar'},
    'PRO': {'charge': 0, 'hydrophobicity': -1.6, 'volume': 112.7, 'polarity': 'nonpolar'}, 'GLN': {'charge': 0, 'hydrophobicity': -3.5, 'volume': 143.8, 'polarity': 'polar'},
    'ARG': {'charge': 1, 'hydrophobicity': -4.5, 'volume': 173.4, 'polarity': 'basic'}, 'SER': {'charge': 0, 'hydrophobicity': -0.8, 'volume': 89.0, 'polarity': 'polar'},
    'THR': {'charge': 0, 'hydrophobicity': -0.7, 'volume': 116.1, 'polarity': 'polar'}, 'VAL': {'charge': 0, 'hydrophobicity': 4.2, 'volume': 140.0, 'polarity': 'nonpolar'},
    'TRP': {'charge': 0, 'hydrophobicity': -0.9, 'volume': 227.8, 'polarity': 'nonpolar'}, 'TYR': {'charge': 0, 'hydrophobicity': -1.3, 'volume': 193.6, 'polarity': 'polar'}
}
POLARITY_MAP = {'nonpolar': 0, 'polar': 1, 'acidic': 2, 'basic': 3}
THREE_TO_ONE_MAP = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
AA_CODES = sorted(AA_PROPERTIES.keys())
AA_MAP = {name: i for i, name in enumerate(AA_CODES)}
V_HEAVY_ONEHOT_COLS = ["IGHV1", "IGHV3", "IGHV5", "IGHV4", "IGHV8", "IGHV7", "IGHV6", "IGHV9", "IGHV14", "unknown_VH"]
V_LIGHT_ONEHOT_COLS = ["IGKV1", "IGLV2", "IGLV3", "IGKV3", "IGKV12", "IGKV4", "IGLV1", "IGKV5", "IGKV8", "IGKV10", "IGKV6", "IGKV1D", "IGKV16", "unknown_VL"]

def get_sabdab_info(sabdab_df, pdb_id):
    entry = sabdab_df[sabdab_df['pdb'].str.lower() == pdb_id.lower()]
    if entry.empty: raise ValueError(f"PDB ID {pdb_id} not found in SAbDab summary.")
    
    # Extract V-gene info if available
    vh_family = None
    vl_family = None
    if 'H_gene' in entry.columns and pd.notna(entry.iloc[0]['H_gene']):
        vh_family = entry.iloc[0]['H_gene'].split('-')[0].split('*')[0]
    if 'L_gene' in entry.columns and pd.notna(entry.iloc[0]['L_gene']):
        vl_family = entry.iloc[0]['L_gene'].split('-')[0].split('*')[0]
        
    return {
        'Hchain': entry.iloc[0]['Hchain'], 'Lchain': entry.iloc[0]['Lchain'],
        'antigen_chain': entry.iloc[0]['antigen_chain'],
        'vh_family': vh_family, 'vl_family': vl_family
    }

def get_chain_sequence(chain):
    """Extracts amino acid sequence from a Bio.PDB chain object."""
    seq = ""
    for residue in chain.get_residues():
        if is_aa(residue, standard=True):
            seq += THREE_TO_ONE_MAP.get(residue.get_resname().upper(), 'X')
    return seq

def extract_cdrs_with_anarci(chain_id, sequence, chain_type):
    """Runs ANARCI to get CDR sequences for a given chain sequence."""
    cdrs = {'cdr1_seq': '', 'cdr2_seq': '', 'cdr3_seq': ''}
    if not sequence: return cdrs

    with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix=".fasta") as tmp_fasta:
        tmp_fasta.write(f">{chain_id}\n{sequence}\n")
        tmp_fasta.flush()
        
        cmd = ["ANARCI", "-i", tmp_fasta.name, "-s", "imgt"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return cdrs # ANARCI failed, not found, or timed out

    lines = result.stdout.splitlines()
    cdr1, cdr2, cdr3 = [], [], []
    
    # CORRECT IMGT CDR definitions (inclusive ranges)
    cdr1_range, cdr2_range, cdr3_range = range(27, 39), range(56, 66), range(105, 118)

    for line in lines:
        parts = line.split()
        # Robustly parse lines that start with H/L, have a position, and an amino acid
        if line.startswith(( 'H', 'L' )) and len(parts) >= 3 and parts[1].isdigit():
            pos, aa = int(parts[1]), parts[2]
            if aa == '-': continue

            if pos in cdr1_range: cdr1.append(aa)
            if pos in cdr2_range: cdr2.append(aa)
            if pos in cdr3_range: cdr3.append(aa)
            
    cdrs['cdr1_seq'] = "".join(cdr1)
    cdrs['cdr2_seq'] = "".join(cdr2)
    cdrs['cdr3_seq'] = "".join(cdr3)
    return cdrs

def process_pdb(pdb_id, pdb_file, output_dir, sabdab_info):
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    # --- Feature Extraction (No changes needed here) ---
    node_features, antigen_residues, antibody_residues = [], [], []
    sasa_structure = freesasa.Structure(pdb_file)
    sasa_result = freesasa.calc(sasa_structure)
    try: dssp = DSSP(model, pdb_file, dssp='mkdssp')
    except Exception: dssp = {}
    antigen_chains = sabdab_info['antigen_chain'].split('/')
    for chain in model:
        if chain.id not in antigen_chains: continue
        for residue in chain:
            if not is_aa(residue, standard=True) or 'CA' not in residue: continue
            res_name, res_id, ca_atom = residue.get_resname(), residue.get_id()[1], residue['CA']
            try: sasa = sasa_result.residueAreas()[chain.id][res_id]['total']
            except KeyError: sasa = 0.0
            dssp_key = (chain.id, (' ', res_id, ' '))
            phi, psi = (dssp[dssp_key][4], dssp[dssp_key][5]) if dssp_key in dssp else (360.0, 360.0)
            props = AA_PROPERTIES.get(res_name, {})
            res_short = THREE_TO_ONE_MAP.get(res_name.upper(), 'X')
            node_features.append({'resId': res_id, 'resName': res_name, 'resShort': res_short, 'bFactor': ca_atom.get_bfactor(), 'sasa': sasa, 'phi': phi, 'psi': psi, 'charge': props.get('charge', 0), 'hydrophobicity': props.get('hydrophobicity', 0.0), 'polarity': POLARITY_MAP.get(props.get('polarity', 'nonpolar'), 0), 'volume': props.get('volume', 0.0)})
            antigen_residues.append(residue)
    if not node_features: return
    rd = ResidueDepth(model)
    for i, residue in enumerate(antigen_residues):
        try: depth = rd[residue.get_parent().id, residue.get_id()][0]
        except (KeyError, IndexError): depth = 0.0
        node_features[i]['resDepth'] = depth
    node_feature_df = pd.DataFrame(node_features)
    aa_dummies = pd.get_dummies(node_feature_df['resName'], prefix='is')
    for col in [f'is{name}' for name in AA_CODES]:
        if col not in aa_dummies.columns: aa_dummies[col] = 0
    node_feature_df = pd.concat([node_feature_df, aa_dummies], axis=1)
    for col in V_HEAVY_ONEHOT_COLS + V_LIGHT_ONEHOT_COLS: node_feature_df[col] = 0
    vh_fam, vl_fam = sabdab_info.get('vh_family'), sabdab_info.get('vl_family')
    if vh_fam and vh_fam in node_feature_df.columns: node_feature_df[vh_fam] = 1
    else: node_feature_df['unknown_VH'] = 1
    if vl_fam and vl_fam in node_feature_df.columns: node_feature_df[vl_fam] = 1
    else: node_feature_df['unknown_VL'] = 1
    for chain_id_str in [sabdab_info['Hchain'], sabdab_info['Lchain']]:
        if pd.isna(chain_id_str): continue
        for chain_id in chain_id_str.split('/'):
            if chain_id in model:
                for residue in model[chain_id]:
                    if is_aa(residue, standard=True) and 'CA' in residue: antibody_residues.append(residue)
    antigen_coords = np.array([res['CA'].get_coord() for res in antigen_residues])
    antibody_coords = np.array([res['CA'].get_coord() for res in antibody_residues])
    if antibody_coords.shape[0] == 0: node_feature_df['isInterface'] = 0
    else:
        kdtree = KDTree(antibody_coords)
        distances, _ = kdtree.query(antigen_coords, k=1)
        node_feature_df['isInterface'] = (distances < 4.5).astype(int)
    node_label_df = node_feature_df[['isInterface']].copy()
    kdtree_antigen = KDTree(antigen_coords)
    edge_pairs = list(kdtree_antigen.query_pairs(r=10.0))
    if not edge_pairs:
        edge_index_df, edge_attr_dist_df, edge_attr_charge_df = pd.DataFrame(columns=['source', 'target']), pd.DataFrame(columns=['dist']), pd.DataFrame(columns=['charge_product'])
    else:
        edge_index_df = pd.DataFrame(edge_pairs, columns=['source', 'target'])
        source_coords, target_coords = antigen_coords[edge_index_df['source']], antigen_coords[edge_index_df['target']]
        edge_attr_dist_df = pd.DataFrame(np.linalg.norm(source_coords - target_coords, axis=1), columns=['dist'])
        source_charges, target_charges = node_feature_df.iloc[edge_index_df['source']]['charge'].values, node_feature_df.iloc[edge_index_df['target']]['charge'].values
        edge_attr_charge_df = pd.DataFrame(source_charges * target_charges, columns=['charge_product'])
    
    # --- File Saving (CDR part is now corrected) ---
    pdb_output_dir = os.path.join(output_dir, pdb_id)
    os.makedirs(pdb_output_dir, exist_ok=True)
    cols_to_drop = ['isInterface', 'resName']
    node_feature_df.drop(columns=[col for col in cols_to_drop if col in node_feature_df.columns]).to_parquet(os.path.join(pdb_output_dir, 'node_feature.parquet'))
    node_label_df.to_parquet(os.path.join(pdb_output_dir, 'node_label_pi.parquet'))
    edge_index_df.to_parquet(os.path.join(pdb_output_dir, 'edge_index.parquet'))
    edge_attr_dist_df.to_parquet(os.path.join(pdb_output_dir, 'edge_attribute_dist.parquet'))
    edge_attr_charge_df.to_parquet(os.path.join(pdb_output_dir, 'edge_attribute_charge.parquet'))

    seq_dir = os.path.join(pdb_output_dir, 'sequence')
    os.makedirs(seq_dir, exist_ok=True)
    antigen_sequence = "".join(node_feature_df['resShort'])
    with open(os.path.join(seq_dir, 'antigen_sequence.json'), 'w') as f:
        json.dump({"pdb_sequence": antigen_sequence}, f)
    
    # CORRECTED: Extract and save real CDRs
    h_chain_id = sabdab_info['Hchain'].split('/')[0] if pd.notna(sabdab_info['Hchain']) else None
    l_chain_id = sabdab_info['Lchain'].split('/')[0] if pd.notna(sabdab_info['Lchain']) else None
    
    h_chain_seq = get_chain_sequence(model[h_chain_id]) if h_chain_id and h_chain_id in model else ""
    l_chain_seq = get_chain_sequence(model[l_chain_id]) if l_chain_id and l_chain_id in model else ""

    h_cdrs = extract_cdrs_with_anarci(h_chain_id, h_chain_seq, 'H')
    l_cdrs = extract_cdrs_with_anarci(l_chain_id, l_chain_seq, 'L')

    final_cdr_data = {
        "H1_seq": h_cdrs.get('cdr1_seq', ''), "H2_seq": h_cdrs.get('cdr2_seq', ''), "H3_seq": h_cdrs.get('cdr3_seq', ''),
        "L1_seq": l_cdrs.get('cdr1_seq', ''), "L2_seq": l_cdrs.get('cdr2_seq', ''), "L3_seq": l_cdrs.get('cdr3_seq', '')
    }
    with open(os.path.join(seq_dir, 'cdr_sequence.json'), 'w') as f:
        json.dump(final_cdr_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDB files to generate graph features.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing raw PDB files.")
    parser.add_argument("--sabdab_summary", type=str, required=True, help="Path to the SAbDab summary TSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed .parquet files.")
    args = parser.parse_args()

    print("Loading SAbDab summary...")
    sabdab_df = pd.read_csv(args.sabdab_summary, sep='\t', low_memory=False)
    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files to process.")
    
    for filename in tqdm(pdb_files, desc="Processing PDBs"):
        pdb_id = os.path.splitext(filename)[0]
        
        pdb_output_dir = os.path.join(args.output_dir, pdb_id)
        final_file_check = os.path.join(pdb_output_dir, 'sequence', 'cdr_sequence.json')
        if os.path.exists(final_file_check):
            try:
                with open(final_file_check, 'r') as f: content = json.load(f)
                if "note" not in content: continue
            except (json.JSONDecodeError, TypeError): pass

        pdb_file_path = os.path.join(args.pdb_dir, filename)
        try:
            sabdab_info = get_sabdab_info(sabdab_df, pdb_id)
            process_pdb(pdb_id, pdb_file_path, args.output_dir, sabdab_info)
        except Exception as e:
            print(f"Could not process {pdb_id}. Error: {e}")

    print("\nPreprocessing complete.")