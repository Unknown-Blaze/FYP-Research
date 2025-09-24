######################################################################################88

FREDHUTCH_HACKS = False # silly stuff Phil added for running on Hutch servers
if FREDHUTCH_HACKS:
    import os
    from shutil import which
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = '1'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '2.0'
    assert which('ptxas') is not None

import argparse

parser = argparse.ArgumentParser(
    description="Run template-based alphafold inference with a fine-tuned regression head",
    epilog = f'''
Examples:

# Run prediction with a fine-tuned regression model
python3 run_prediction.py \\
    --targets path/to/your/targets_for_prediction.csv \\
    --data_dir /path/to/alphafold/data/ \\
    --outfile_prefix my_regression_predictions \\
    --model_names model_2_ptm_regression \\
    --model_params_files /path/to/your/trained_model.pkl

    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)

parser.add_argument('--outfile_prefix',
                    help='Prefix that will be prepended to the output '
                    'filenames')
parser.add_argument('--final_outfile_prefix',
                    help='Prefix that will be prepended to the final output '
                    'tsv filename')
parser.add_argument('--targets', required=True, help='File listing the targets to '
                    'be modeled. See description of file format in the github '
                    'README and also examples in the examples/*/*tsv')
parser.add_argument('--data_dir', help='Location of AlphaFold params/ folder')

parser.add_argument('--model_names', type=str, nargs='*', default=['model_2_ptm'])
parser.add_argument('--model_params_files', type=str, nargs='*')
parser.add_argument('--output_dir',
                    help='Directory where the prediction files will be saved',
                    default='.')
                    
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--ignore_identities', action='store_true',
                    help='Ignore the sequence identities column in the templates '
                    'alignment files. Useful when modeling many different peptides '
                    'using the same alignment file.')
parser.add_argument('--no_pdbs', action='store_true', help='Dont write out pdbs')
parser.add_argument('--terse', action='store_true', help='Dont write out pdbs or '
                    'matrices with alphafold confidence values')
parser.add_argument('--no_resample_msa', action='store_true', help='Dont randomly '
                    'resample from the MSA during recycling. Perhaps useful for '
                    'testing...')

args = parser.parse_args()

import os
os.makedirs(args.output_dir, exist_ok=True)

import sys
from os.path import exists
import itertools
import numpy as np
import pandas as pd
import predict_utils
import jax
import jax.numpy as jnp
import haiku as hk

# --- UPDATED CLASS TO MATCH IMPROVED FINE-TUNING ---
import train_utils # Contains helper functions like compute_predicted_aligned_error_jax

class PaeBinderRegressor(hk.Module):
    """
    Improved regressor that predicts pIC50 using PAE, peptide length, and MHC class.
    This MUST match the architecture used in fine-tuning!
    Features: proper normalization, dropout, and hidden layers.
    """
    def __init__(self, num_mhc_classes=2, dropout_rate=0.1, *args, **kwargs):
        super().__init__(name="PaeBinderRegressor")
        self.num_mhc_classes = num_mhc_classes
        self.dropout_rate = dropout_rate

    def __call__(self, input_features, is_training=False):  # Default to inference mode
        # Extract features from the input dictionary
        mask1 = input_features['partner1_mask']
        mask2 = input_features['partner2_mask']
        mhc_class_1hot = input_features['mhc_class_1hot']
        peptide_len = input_features['peptide_len']

        # Calculate PAE feature from AlphaFold's output
        # Handle both dictionary and direct array cases
        if isinstance(input_features['predicted_aligned_error'], dict):
            # New format with logits and breaks
            paes = train_utils.compute_predicted_aligned_error_jax(
                input_features['predicted_aligned_error']['logits'],
                input_features['predicted_aligned_error']['breaks'].ravel(),
            )['predicted_aligned_error']
        else:
            # Direct PAE array (already computed)
            paes = input_features['predicted_aligned_error']

        # Compute cross-interface PAE (improved calculation)
        paes12 = jnp.sum((paes * mask1[..., None]) * mask2[..., None, :])
        paes21 = jnp.sum((paes * mask2[..., None]) * mask1[..., None, :])
        total_pairs = jnp.sum(mask1) * jnp.sum(mask2)
        
        # Average PAE between partners, with better scaling
        pae_feature_raw = (paes12 + paes21) / (2 * total_pairs + 1e-8)  # Add epsilon for numerical stability
        
        # Normalize peptide length (center around typical peptide lengths)
        len_feature_raw = (peptide_len - 9.0) / 4.0  # Adjusted normalization
        
        # Ensure proper shape and apply feature normalization
        pae_feature = pae_feature_raw.reshape(-1, 1)
        len_feature = len_feature_raw.reshape(-1, 1)
        
        # Apply feature normalization (same as training)
        pae_feature = (pae_feature - 8.0) / 4.0  # Typical PAE range is 4-16
        len_feature = (len_feature - 0.0) / 1.0   # Already normalized above
        
        # Combine features
        features = jnp.concatenate([pae_feature, len_feature], axis=-1)
        
        # Add additional features for better prediction
        # Calculate interface confidence (complement of PAE)
        interface_confidence = jnp.exp(-pae_feature_raw / 4.0).reshape(-1, 1)  # Convert PAE to confidence
        features = jnp.concatenate([features, interface_confidence], axis=-1)
        
        # Improved neural network architecture (MUST match training)
        num_features = features.shape[-1]
        
        # Hidden layer with more capacity
        hidden_dim = 16
        w1_init = hk.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        b1_init = hk.initializers.Constant(0.0)
        
        w1 = hk.get_parameter("hidden_weights", shape=[num_features, hidden_dim], dtype=features.dtype, init=w1_init)
        b1 = hk.get_parameter("hidden_biases", shape=[hidden_dim], dtype=features.dtype, init=b1_init)
        
        # Hidden layer with activation
        hidden = features @ w1 + b1
        hidden = jax.nn.tanh(hidden)  # Use tanh for better gradient flow
        
        # No dropout during inference (is_training=False by default)
        if is_training and self.dropout_rate > 0:
            key = hk.next_rng_key()
            hidden = hk.dropout(key, self.dropout_rate, hidden)
        
        # Output layer - separate models for each MHC class
        w2_init = hk.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
        # Better initialization based on expected pIC50 values
        b2_init = hk.initializers.Constant(jnp.array([6.0, 5.0]))  # Typical pIC50 values for Class I and II
        
        weights = hk.get_parameter("output_weights", shape=[hidden_dim, self.num_mhc_classes], dtype=features.dtype, init=w2_init)
        biases = hk.get_parameter("output_biases", shape=[self.num_mhc_classes], dtype=features.dtype, init=b2_init)
        
        # Calculate prediction for all classes
        prediction_per_class = hidden @ weights + biases
        
        # Use the one-hot encoding to select the prediction from the correct MHC class
        predicted_pic50 = jnp.sum(prediction_per_class * mhc_class_1hot, axis=-1, keepdims=True)
        
        return predicted_pic50, features

def binder_regression_fn(input_features, is_training=False):
    model = PaeBinderRegressor(num_mhc_classes=2, dropout_rate=0.1)(input_features, is_training=is_training)
    return model

binder_regressor = hk.transform(binder_regression_fn, apply_rng=True)
# --- END UPDATED CLASS ---

targets = pd.read_csv(args.targets) # Use read_csv for safety with your new format

lens = [len(x.target_chainseq.replace('/',''))
        for x in targets.itertuples()]
crop_size = max(lens)

if args.verbose:
    from os import popen
    platform = jax.local_devices()[0].platform
    hostname = popen('hostname').readlines()[0].strip()
    print('cmd:', ' '.join(sys.argv))
    print('local_device:', platform, 'hostname:', hostname, 'num_targets:',
          targets.shape[0], 'max_len=', crop_size)

sys.stdout.flush()

model_runners = predict_utils.load_model_runners(
    args.model_names,
    crop_size,
    args.data_dir,
    model_params_files=args.model_params_files,
    resample_msa_in_recycling = not args.no_resample_msa,
)

final_dfl = []
for counter, targetl in targets.iterrows():
    print('START:', counter, 'of', targets.shape[0])

    alignfile = targetl.templates_alignfile
    assert exists(alignfile)

    query_chainseq = targetl.target_chainseq
    if 'outfile_prefix' in targetl:
        outfile_prefix = targetl.outfile_prefix
    else:
        assert args.outfile_prefix is not None
        if 'targetid' in targetl:
            outfile_prefix = args.outfile_prefix+'_'+targetl.targetid
        else:
            outfile_prefix = f'{args.outfile_prefix}_T{counter}'

    query_sequence = query_chainseq.replace('/','')
    num_res = len(query_sequence)

    data = pd.read_table(alignfile)
    template_features_list = []
    for tnum, row in data.iterrows():
        assert row.target_len == len(query_sequence)
        target_to_template_alignment = {
            int(x.split(':')[0]) : int(x.split(':')[1])
            for x in row.target_to_template_alignstring.split(';')
        }
        template_features = predict_utils.create_single_template_features(
            query_sequence, row.template_pdbfile, target_to_template_alignment,
            f'T{tnum:03d}', allow_chainbreaks=True, allow_skipped_lines=True,
            expected_identities = None if args.ignore_identities else row.identities,
            expected_template_len = row.template_len,
        )
        template_features_list.append(template_features)

    all_template_features = predict_utils.compile_template_features(
        template_features_list)

    msa=[query_sequence]
    deletion_matrix=[[0]*len(query_sequence)]

    # This is the corrected call
    all_metrics, all_outputs = predict_utils.run_alphafold_prediction( # Modified to get all_outputs
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        # Combine the directory and prefix into a full path
        out_prefix=os.path.join(args.output_dir, outfile_prefix), # <-- CORRECTED LINE
        crop_size=crop_size,
        dump_pdbs = not (args.no_pdbs or args.terse),
        dump_metrics = not args.terse,
    )

    outl = targetl.copy()
    for model_name, metrics in all_metrics.items():
        plddts = metrics['plddt']
        paes = metrics.get('predicted_aligned_error', None)

        # --- UPDATED REGRESSION PREDICTION ---
        # This section runs the improved regression head
        
        # 1. Get the full model parameters for the current model
        model_runner = model_runners[model_name]
        params = model_runner.params
        
        # 2. Partition the parameters to isolate the ones for our regressor
        binder_params, _ = hk.data_structures.partition(
            lambda m, n, p: m[:9] != "alphafold", params)

        # 3. Check if binder parameters exist. If not, we can't make a prediction.
        if binder_params:
            try:
                # 4. Prepare the input dictionary for the regressor
                regressor_dict = {}
                
                # Get features from the main AlphaFold output
                predicted_dict = all_outputs[model_name]
                print(f"  Available keys in predicted_dict: {list(predicted_dict.keys())}")
                print(f"  PAE structure: {type(predicted_dict['predicted_aligned_error'])}")
                if isinstance(predicted_dict['predicted_aligned_error'], dict):
                    print(f"  PAE dict keys: {list(predicted_dict['predicted_aligned_error'].keys())}")
                regressor_dict['predicted_aligned_error'] = predicted_dict['predicted_aligned_error']

                # Get other features from the input target file
                peptide_positions_str = targetl.peptide_positions
                print(f"  Raw peptide_positions: {peptide_positions_str} (type: {type(peptide_positions_str)})")
                
                # Parse peptide positions more carefully
                if isinstance(peptide_positions_str, str):
                    peptide_positions_raw = [int(x) for x in peptide_positions_str.split(';')]
                else:
                    # Handle case where it might be a number or other type
                    peptide_positions_raw = [int(peptide_positions_str)]
                
                # Convert from 1-indexed to 0-indexed (subtract 1 from each position)
                peptide_positions = [max(0, x - 1) for x in peptide_positions_raw]
                print(f"  Converted 1-indexed positions {peptide_positions_raw} to 0-indexed {peptide_positions}")
                
                # Validate all positions are within bounds
                valid_positions = [x for x in peptide_positions if 0 <= x < num_res]
                if len(valid_positions) != len(peptide_positions):
                    print(f"  Warning: Some positions were out of bounds. Using {valid_positions}")
                    peptide_positions = valid_positions
                
                nonpeptide_positions = [x for x in range(num_res) if x not in peptide_positions]
                print(f"  Final peptide_positions: {peptide_positions}")
                print(f"  nonpeptide_positions: {len(nonpeptide_positions)} positions")
                print(f"  num_res: {num_res}, crop_size: {crop_size}")
                
                regressor_dict['partner1_mask'] = np.zeros((crop_size,), dtype=np.float32)
                regressor_dict['partner1_mask'][nonpeptide_positions] = 1.0
                regressor_dict['partner2_mask'] = np.zeros((crop_size,), dtype=np.float32)
                regressor_dict['partner2_mask'][peptide_positions] = 1.0
                print(f"  partner1_mask sum: {np.sum(regressor_dict['partner1_mask'])}")
                print(f"  partner2_mask sum: {np.sum(regressor_dict['partner2_mask'])}")

                # MHC class handling with better error checking
                mhc_class_map = {'I': 0, 'II': 1}
                if hasattr(targetl, 'iedb_mhc_class') and pd.notna(targetl.iedb_mhc_class):
                    class_str = str(targetl.iedb_mhc_class).upper()
                    binder_class_int = mhc_class_map.get(class_str, 0)
                else:
                    binder_class_int = 0  # Default to Class I
                
                regressor_dict['mhc_class_1hot'] = np.eye(2, dtype=np.float32)[np.array([binder_class_int])]
                print(f"  MHC class: {binder_class_int}, one-hot: {regressor_dict['mhc_class_1hot']}")
                
                # Peptide length handling with better error checking
                if hasattr(targetl, 'iedb_peptide_len') and pd.notna(targetl.iedb_peptide_len):
                    peptide_len = float(targetl.iedb_peptide_len)
                else:
                    # Calculate from peptide_positions
                    peptide_len = len(peptide_positions)
                
                regressor_dict['peptide_len'] = np.array([[peptide_len]], dtype=np.float32)
                print(f"  Peptide length: {peptide_len}")

                # 5. Run the regressor model (inference mode)
                # We need a dummy JAX random key
                rng = jax.random.PRNGKey(0)
                print(f"  About to run regressor.apply...")
                predicted_pic50, features = binder_regressor.apply(
                    binder_params, rng, regressor_dict, is_training=False)
                
                # 6. Save the prediction
                pic50_value = float(predicted_pic50[0][0])
                outl[model_name+'_pred_pIC50'] = pic50_value
                
                # Debug: Print the prediction to confirm it's being calculated
                print(f"  Predicted pIC50 for {model_name}: {pic50_value}")
                
                # Also save some diagnostic information
                if args.verbose:
                    pae_val = float(features[0][0])  # PAE feature
                    len_val = float(features[0][1])  # Length feature
                    conf_val = float(features[0][2])  # Interface confidence
                    print(f"  pIC50: {pic50_value:.3f}, PAE_feat: {pae_val:.3f}, Len_feat: {len_val:.3f}, Conf: {conf_val:.3f}")
                    
            except Exception as e:
                print(f"Error running regression head for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                print(f"  Setting pIC50 to NaN due to error")
                outl[model_name+'_pred_pIC50'] = np.nan
        else:
            print(f"Warning: No regression head parameters found in model '{model_name}'. Skipping pIC50 prediction.")
            print(f"  Setting pIC50 to NaN due to missing parameters")
            outl[model_name+'_pred_pIC50'] = np.nan
        # --- END UPDATED REGRESSION PREDICTION ---

        cs = query_chainseq.split('/')
        chain_stops = list(itertools.accumulate(len(x) for x in cs))
        chain_starts = [0]+chain_stops[:-1]
        nres = chain_stops[-1]
        assert nres == num_res
        outl[model_name+'_plddt'] = np.mean(plddts[:nres])
        if paes is not None:
            outl[model_name+'_pae'] = np.mean(paes[:nres,:nres])
        for chain1,(start1,stop1) in enumerate(zip(chain_starts, chain_stops)):
            outl[f'{model_name}_plddt_{chain1}'] = np.mean(plddts[start1:stop1])

            if paes is not None:
                for chain2 in range(len(cs)):
                    start2, stop2 = chain_starts[chain2], chain_stops[chain2]
                    pae = np.mean(paes[start1:stop1,start2:stop2])
                    outl[f'{model_name}_pae_{chain1}_{chain2}'] = pae
    final_dfl.append(outl)

if args.final_outfile_prefix:
    outfile_prefix = args.final_outfile_prefix
elif args.outfile_prefix:
    outfile_prefix = args.outfile_prefix
elif 'outfile_prefix' in targets.columns:
    outfile_prefix = targets.outfile_prefix.iloc[0]
else:
    outfile_prefix = None

if outfile_prefix:
    # Construct the full path by joining the output directory and the filename
    outfile = os.path.join(args.output_dir, f'{outfile_prefix}_final.tsv')
    
    # Create DataFrame more robustly to ensure all columns are preserved
    final_df = pd.DataFrame(final_dfl)
    
    # Debug: Print column info
    print(f"Final DataFrame shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    
    # Check for any pIC50 columns
    pic50_cols = [col for col in final_df.columns if 'pred_pIC50' in col]
    print(f"pIC50 prediction columns: {pic50_cols}")
    
    for col in pic50_cols:
        non_null_count = final_df[col].notna().sum()
        print(f"  {col}: {non_null_count} non-null values out of {len(final_df)}")
    
    final_df.to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)