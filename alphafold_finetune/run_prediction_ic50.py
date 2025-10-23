######################################################################################
# Enhanced AlphaFold Fine-tuning Prediction Script
# Run template-based alphafold inference with a fine-tuned regression head

FREDHUTCH_HACKS = False # silly stuff Phil added for running on Hutch servers
if FREDHUTCH_HACKS:
    import os
    from shutil import which
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = '1'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '2.0'
    assert which('ptxas') is not None

import argparse
import os
import sys
from os.path import exists
import itertools
import numpy as np
import pandas as pd
import predict_utils
import jax
import jax.numpy as jnp
import haiku as hk
import train_utils  # Contains helper functions like compute_predicted_aligned_error_jax

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

os.makedirs(args.output_dir, exist_ok=True)

# --- ENHANCED REGRESSOR TO MATCH IMPROVED FINE-TUNING ---

class PaeBinderRegressor(hk.Module):
    """
    Enhanced regressor that matches the improved training architecture.
    Features: multi-layer network, attention mechanism, comprehensive features.
    """
    def __init__(self, num_mhc_classes=2, dropout_rate=0.15, hidden_dims=[32, 16], use_attention=True, *args, **kwargs):
        super().__init__(name="PaeBinderRegressor")
        self.num_mhc_classes = num_mhc_classes
        self.dropout_rate = dropout_rate
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention

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
            # Legacy format - direct PAE array
            paes = input_features['predicted_aligned_error']

        # Enhanced interface analysis with multiple PAE metrics (matching training)
        interface_mask = mask1[..., None] * mask2[..., None, :]
        interface_paes = paes * interface_mask
        
        # Basic cross-interface PAE
        paes12 = jnp.sum((paes * mask1[..., None]) * mask2[..., None, :])
        paes21 = jnp.sum((paes * mask2[..., None]) * mask1[..., None, :])
        total_pairs = jnp.sum(mask1) * jnp.sum(mask2)
        avg_interface_pae = (paes12 + paes21) / (2 * total_pairs + 1e-8)
        
        # Additional PAE statistics for richer features - JAX-compatible version
        # Use jnp.where instead of boolean indexing for JIT compilation
        interface_paes_flat = interface_paes.ravel()
        interface_mask_flat = interface_mask.ravel()
        
        # JAX-safe way to compute statistics over masked values
        masked_paes = jnp.where(interface_mask_flat > 0, interface_paes_flat, jnp.inf)
        n_valid = jnp.sum(interface_mask_flat > 0)
        
        # Compute min/max safely (use inf for invalid entries)
        min_interface_pae = jnp.where(n_valid > 0, 
                                    jnp.min(masked_paes), 
                                    avg_interface_pae)
        
        # For max, use -inf for invalid entries  
        masked_paes_max = jnp.where(interface_mask_flat > 0, interface_paes_flat, -jnp.inf)
        max_interface_pae = jnp.where(n_valid > 0, 
                                    jnp.max(masked_paes_max), 
                                    avg_interface_pae)
        
        # For std, compute mean first then std manually
        valid_mean = jnp.where(n_valid > 0, 
                             jnp.sum(jnp.where(interface_mask_flat > 0, interface_paes_flat, 0.0)) / n_valid,
                             avg_interface_pae)
        valid_var = jnp.where(n_valid > 1,
                            jnp.sum(jnp.where(interface_mask_flat > 0, 
                                            (interface_paes_flat - valid_mean) ** 2, 0.0)) / n_valid,
                            0.0)
        std_interface_pae = jnp.sqrt(valid_var)
        
        # Confidence features (complement of PAE)
        interface_confidence = jnp.exp(-avg_interface_pae / 4.0)
        min_confidence = jnp.exp(-max_interface_pae / 4.0)
        
        # Sequence-based features with better normalization
        len_feature_raw = (peptide_len - 9.0) / 4.0  # Center around typical peptide lengths
        len_squared = len_feature_raw ** 2  # Non-linear length effect
        
        # Structural quality features from pLDDT if available
        if 'plddt' in input_features:
            plddt = input_features['plddt']
            partner1_plddt = jnp.sum(plddt * mask1) / (jnp.sum(mask1) + 1e-8)
            partner2_plddt = jnp.sum(plddt * mask2) / (jnp.sum(mask2) + 1e-8)
            avg_plddt = (partner1_plddt + partner2_plddt) / 2.0
        else:
            partner1_plddt = jnp.array(75.0)  # Default confidence
            partner2_plddt = jnp.array(75.0)
            avg_plddt = jnp.array(75.0)
        
        # Normalize all features (same as training)
        avg_interface_pae = (avg_interface_pae - 8.0) / 4.0  # PAE range 4-16
        min_interface_pae = (min_interface_pae - 8.0) / 4.0
        max_interface_pae = (max_interface_pae - 8.0) / 4.0
        std_interface_pae = std_interface_pae / 4.0
        len_feature_raw = len_feature_raw / 1.0  # Already normalized
        partner1_plddt = (partner1_plddt - 70.0) / 20.0  # pLDDT range normalization
        partner2_plddt = (partner2_plddt - 70.0) / 20.0
        avg_plddt = (avg_plddt - 70.0) / 20.0
        
        # Combine all features into a comprehensive feature vector (same as training)
        all_features = jnp.array([
            avg_interface_pae.ravel()[0],
            min_interface_pae.ravel()[0],
            max_interface_pae.ravel()[0], 
            std_interface_pae.ravel()[0],
            interface_confidence.ravel()[0],
            min_confidence.ravel()[0],
            len_feature_raw.ravel()[0],
            len_squared.ravel()[0],
            partner1_plddt.ravel()[0],
            partner2_plddt.ravel()[0],
            avg_plddt.ravel()[0]
        ]).reshape(1, -1)

        # Advanced multi-layer neural network (matching training architecture)
        x = all_features
        num_features = x.shape[-1]
        
        # Apply feature attention if enabled
        if self.use_attention:
            attention_weights = hk.Linear(num_features, name='attention_weights')(x)
            attention_weights = jax.nn.softmax(attention_weights, axis=-1)
            x = x * attention_weights  # Weighted feature importance
        
        # Multi-layer architecture with residual connections
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Store input for potential residual connection
            residual = x if x.shape[-1] == hidden_dim else None
            
            # Linear transformation with proper initialization
            w_init = hk.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            x = hk.Linear(hidden_dim, w_init=w_init, name=f'hidden_{i}')(x)
            
            # Layer normalization for stable training
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f'ln_{i}')(x)
            
            # Advanced activation function (Swish/SiLU performs better than tanh)
            x = jax.nn.swish(x)
            
            # Residual connection if dimensions match
            if residual is not None:
                x = x + 0.1 * residual  # Scaled residual for stability
            
            # No dropout during inference (is_training=False by default)
            if is_training and self.dropout_rate > 0:
                key = hk.next_rng_key()
                x = hk.dropout(key, self.dropout_rate, x)
        
        # MHC class-specific output heads with better initialization
        class_predictions = []
        for mhc_class in range(self.num_mhc_classes):
            # Separate head for each MHC class
            class_head = hk.Linear(1, name=f'output_class_{mhc_class}')
            class_pred = class_head(x)
            class_predictions.append(class_pred)
        
        # Combine class-specific predictions
        prediction_per_class = jnp.concatenate(class_predictions, axis=-1)
        
        # Apply class-specific bias based on typical pIC50 values
        class_bias = jnp.array([6.0, 5.0])  # Class I typically higher than Class II
        prediction_per_class = prediction_per_class + class_bias
        
        # Select prediction based on MHC class
        predicted_pic50 = jnp.sum(prediction_per_class * mhc_class_1hot, axis=-1, keepdims=True)
        
        return predicted_pic50, all_features

def binder_regression_fn(input_features, is_training=False):
    model = PaeBinderRegressor(
        num_mhc_classes=2, 
        dropout_rate=0.15,
        hidden_dims=[32, 16], 
        use_attention=True
    )(input_features, is_training=is_training)
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