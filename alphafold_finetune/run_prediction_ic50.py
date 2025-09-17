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
import sys
from os.path import exists
import itertools
import numpy as np
import pandas as pd
import predict_utils
import jax
import haiku as hk

# --- CHANGE START ---
# We need to add the PaeBinderRegressor class and its dependencies,
# as it's required to run our fine-tuned head.
import train_utils # Contains helper functions like compute_predicted_aligned_error_jax

class PaeBinderRegressor(hk.Module):
    """
    Predicts pIC50 using PAE, peptide length, and MHC class.
    (This class is copied directly from the fine-tuning script)
    """
    def __init__(self, num_mhc_classes=2, *args, **kwargs):
        super().__init__(name="PaeBinderRegressor")
        self.num_mhc_classes = num_mhc_classes

    def __call__(self, input_features):
        mask1 = input_features['partner1_mask']
        mask2 = input_features['partner2_mask']
        mhc_class_1hot = input_features['mhc_class_1hot']
        peptide_len = input_features['peptide_len']

        paes = train_utils.compute_predicted_aligned_error_jax(
            input_features['predicted_aligned_error']['logits'],
            input_features['predicted_aligned_error']['breaks'].ravel(),
        )['predicted_aligned_error']

        paes12 = jnp.sum((paes * mask1[..., None]) * mask2[..., None, :], keepdims=True)
        paes21 = jnp.sum((paes * mask2[..., None]) * mask1[..., None, :], keepdims=True)
        
        pae_feature = 0.1 * (paes12 + paes21)[0] / (2 * jnp.sum(mask1) * jnp.sum(mask2))
        len_feature = (peptide_len - 9.0) / 3.0
        features = jnp.concatenate([pae_feature, len_feature], axis=-1)

        num_features = features.shape[-1]
        w_init = hk.initializers.RandomNormal(stddev=0.02)
        b_init = hk.initializers.Constant(jnp.array([5.5, 4.5])) 

        weights = hk.get_parameter("weights", shape=[num_features, self.num_mhc_classes], dtype=features.dtype, init=w_init)
        biases = hk.get_parameter("biases", shape=[self.num_mhc_classes], dtype=features.dtype, init=b_init)

        prediction_per_class = features @ weights + biases
        predicted_pic50 = jnp.sum(prediction_per_class * mhc_class_1hot, axis=-1, keepdims=True)

        return predicted_pic50, features

def binder_regression_fn(input_features):
    model = PaeBinderRegressor(num_mhc_classes=2)(input_features)
    return model

binder_regressor = hk.transform(binder_regression_fn, apply_rng=True)
# --- CHANGE END ---


targets = pd.read_csv(args.targets) # Use read_csv for safety with your new format

lens = [len(x.target_chainseq.replace('/',''))
        for x in targets.it_tuples()]
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

    all_metrics, all_outputs = predict_utils.run_alphafold_prediction( # Modified to get all_outputs
        query_sequence=query_sequence,
        msa=msa,
        deletion_matrix=deletion_matrix,
        chainbreak_sequence=query_chainseq,
        template_features=all_template_features,
        model_runners=model_runners,
        out_prefix=outfile_prefix,
        crop_size=crop_size,
        dump_pdbs = not (args.no_pdbs or args.terse),
        dump_metrics = not args.terse,
    )

    outl = targetl.copy()
    for model_name, metrics in all_metrics.items():
        plddts = metrics['plddt']
        paes = metrics.get('predicted_aligned_error', None)

        # --- CHANGE START ---
        # This is the new section to run the regression head
        
        # 1. Get the full model parameters for the current model
        model_runner = model_runners[model_name]
        params = model_runner.params
        
        # 2. Partition the parameters to isolate the ones for our regressor
        binder_params, _ = hk.data_structures.partition(
            lambda m, n, p: m[:9] != "alphafold", params)

        # 3. Check if binder parameters exist. If not, we can't make a prediction.
        if binder_params:
            # 4. Prepare the input dictionary for the regressor
            regressor_dict = {}
            
            # Get features from the main AlphaFold output
            predicted_dict = all_outputs[model_name]
            regressor_dict['predicted_aligned_error'] = predicted_dict['predicted_aligned_error']

            # Get other features from the input target file
            peptide_positions = [int(x) for x in targetl.peptide_positions.split(';')]
            nonpeptide_positions = [x for x in range(num_res) if x not in peptide_positions]
            
            regressor_dict['partner1_mask'] = np.zeros((crop_size,))
            regressor_dict['partner1_mask'][nonpeptide_positions] = 1.
            regressor_dict['partner2_mask'] = np.zeros((crop_size,))
            regressor_dict['partner2_mask'][peptide_positions] = 1.

            mhc_class_map = {'I': 0, 'II': 1}
            class_str = str(targetl.get('iedb_mhc_class', 'I'))
            binder_class_int = mhc_class_map.get(class_str, 0)
            regressor_dict['mhc_class_1hot'] = np.eye(2)[np.array([binder_class_int])]
            
            regressor_dict['peptide_len'] = np.array([[float(targetl.iedb_peptide_len)]])

            # 5. Run the regressor model
            # We need a dummy JAX random key
            rng = jax.random.PRNGKey(0)
            predicted_pic50, _ = binder_regressor.apply(binder_params, rng, regressor_dict)
            
            # 6. Save the prediction
            outl[model_name+'_pred_pIC50'] = float(predicted_pic50[0][0])
        else:
            print(f"Warning: No regression head parameters found in model '{model_name}'. Skipping pIC50 prediction.")
            outl[model_name+'_pred_pIC50'] = np.nan
        # --- CHANGE END ---

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
    outfile = f'{outfile_prefix}_final.tsv'
    pd.DataFrame(final_dfl).to_csv(outfile, sep='\t', index=False)
    print('made:', outfile)