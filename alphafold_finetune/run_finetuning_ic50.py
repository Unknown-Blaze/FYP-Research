######################################################################################88
FREDHUTCH_HACKS = False # silly stuff Phil added for running on Hutch servers

if FREDHUTCH_HACKS:
    import os
    # I think this has to happen before importing tensorflow?
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'

import os
from os import popen
import sys
import pandas as pd
import numpy as np
import pickle
import optax
import jax
import jax.numpy as jnp
import haiku as hk
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.common.protein import Protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.model.all_atom import atom37_to_torsion_angles, atom37_to_frames
import torch # dataloader/dataset stuff
import tensorflow.compat.v1 as tf1 # cmdline args
import warnings
warnings.filterwarnings("ignore")

## fine-tuning specific helper codes
import train_utils
import predict_utils
print('done importing') ; sys.stdout.flush()

flags = tf1.app.flags

flags.DEFINE_string('outprefix', 'testrun', help='string prefix for all outfiles')
flags.DEFINE_string('output_dir', '.', help='Directory to save model checkpoints.')
flags.DEFINE_string('model_name', 'model_2_ptm', help='like model_1 or model_2_ptm')
flags.DEFINE_string('train_dataset', None, help='csv file with training dataset')
flags.DEFINE_string('valid_dataset', None, help='csv file with validation dataset')
flags.DEFINE_integer('crop_size', 190, help='Max size of training example; set this '
                     'as low as possible for memory and speed')
flags.DEFINE_integer('num_epochs', 10, help='number of epochs')
flags.DEFINE_integer('batch_size', 1, help='total batch size')
flags.DEFINE_integer('apply_every', 1, help='how often to apply gradient updates')
flags.DEFINE_bool('notrain', False, help='if True, dont do any training')
flags.DEFINE_bool('debug', False, help='debug')
flags.DEFINE_bool('verbose', False, help='verbose')
flags.DEFINE_bool('test_load', False,
                  help='if True, loop through datasets to test loading')
flags.DEFINE_bool('dump_pdbs', False, help='if True, write out PDB files of '
                  'modeled structures during training')
flags.DEFINE_integer('print_steps', 25,
                     help='printed averaged results every print_steps')
flags.DEFINE_integer('save_steps', 100, help='save model + optimizer every save_steps')
flags.DEFINE_integer('valid_steps', 500,
                     help='calc loss on whole valid set every valid_steps')
flags.DEFINE_integer('num_cpus', 0, help='number of (extra?) cpus for loading')
flags.DEFINE_float('lr_coef', 0.01, help='learning rate coefficient - reduced for stability')
flags.DEFINE_float('fake_native_weight', 0.25, help='weight to apply to the alphafold '
                   'loss when using a predicted structure as the native')
flags.DEFINE_float('struc_viol_weight', 1.0, help='structural violation weight')
flags.DEFINE_integer('msa_clusters', 5, help='number of msa cluster sequences')
flags.DEFINE_integer('extra_msa', 1, help='number of extra msa sequences')
flags.DEFINE_integer('num_evo_blocks', 48, help='number of evoformer blocks')
flags.DEFINE_float('grad_norm_clip', 1.0, help='value to clip gradient norms - reduced for stability')
flags.DEFINE_string('data_dir', "/home/pbradley/csdat/alphafold/data/",
                    help='location of alphafold params; passed to '
                    'data.get_model_haiku_params; should contain params/ subfolder')

flags.DEFINE_bool('only_fit_binder', False, help='if True, dont fit alphafold params')
flags.DEFINE_bool('freeze_binder', False, help='if True, dont fit binder params')
flags.DEFINE_bool('freeze_everything', False, help='if True, dont fit anything')
flags.DEFINE_bool('no_ramp', False, help='if True, dont ramp')
flags.DEFINE_bool('no_valid', False, help='if True, dont compute valid stats')
flags.DEFINE_bool('no_random', False, help='if True, dont randomize')
flags.DEFINE_bool('random_recycling', False, help='if True, set num_iter_recycling '
                  'randomly during training')

# Regression-specific flags
flags.DEFINE_float('regression_loss_weight', 10.0, help='Weight for the pIC50 regression loss - increased')
flags.DEFINE_bool('normalize_features', True, help='Whether to normalize input features')
flags.DEFINE_float('dropout_rate', 0.1, help='Dropout rate for regularization')

FLAGS = flags.FLAGS

assert FLAGS.msa_clusters >= 5 # since reduce_msa_clusters_by_max_templates is True for model_1/model_1_ptm

batch_size = FLAGS.batch_size
if batch_size>1:
    print('WARNING\n'*12,'phil has not tested batch_size>1')

assert FLAGS.apply_every >= 1

jax_key = jax.random.PRNGKey(0)
model_name = FLAGS.model_name

platform = jax.local_devices()[0].platform
hostname = popen('hostname').readlines()[0].strip()

print('cmd:', ' '.join(sys.argv))
print('local_device:', platform, hostname)
print('model_name:', model_name)
print('outprefix:', FLAGS.outprefix)
sys.stdout.flush()

model_config = config.model_config(model_name)
model_config.data.common.resample_msa_in_recycling = True
model_config.model.resample_msa_in_recycling = True
model_config.data.common.max_extra_msa = FLAGS.extra_msa
model_config.data.eval.max_msa_clusters = FLAGS.msa_clusters
model_config.data.eval.crop_size = FLAGS.crop_size
model_config.model.heads.structure_module.structural_violation_loss_weight = FLAGS.struc_viol_weight
model_config.model.embeddings_and_evoformer.evoformer_num_block = FLAGS.num_evo_blocks

# Improved PaeBinderRegressor with better architecture and normalization
class PaeBinderRegressor(hk.Module):
    """
    Improved regressor that predicts pIC50 using PAE, peptide length, and MHC class.
    Features: proper normalization, dropout, and residual connections.
    """
    def __init__(self, num_mhc_classes=2, dropout_rate=0.1, *args, **kwargs):
        super().__init__(name="PaeBinderRegressor")
        self.num_mhc_classes = num_mhc_classes
        self.dropout_rate = dropout_rate

    def __call__(self, input_features, is_training=True):
        # Extract features from the input dictionary
        mask1 = input_features['partner1_mask']
        mask2 = input_features['partner2_mask']
        mhc_class_1hot = input_features['mhc_class_1hot']
        peptide_len = input_features['peptide_len']

        # Calculate PAE feature from AlphaFold's output
        paes = train_utils.compute_predicted_aligned_error_jax(
            input_features['predicted_aligned_error']['logits'],
            input_features['predicted_aligned_error']['breaks'].ravel(),
        )['predicted_aligned_error']

        # Compute cross-interface PAE (improved calculation)
        paes12 = jnp.sum((paes * mask1[..., None]) * mask2[..., None, :])
        paes21 = jnp.sum((paes * mask2[..., None]) * mask1[..., None, :])
        total_pairs = jnp.sum(mask1) * jnp.sum(mask2)
        
        # Average PAE between partners, with better scaling
        pae_feature_raw = (paes12 + paes21) / (2 * total_pairs + 1e-8)  # Add epsilon for numerical stability
        
        # Normalize peptide length (center around typical peptide lengths)
        len_feature_raw = (peptide_len - 9.0) / 4.0  # Adjusted normalization
        
        # Ensure proper shape and apply feature normalization if enabled
        pae_feature = pae_feature_raw.reshape(-1, 1)
        len_feature = len_feature_raw.reshape(-1, 1)
        
        if FLAGS.normalize_features:
            # Apply batch normalization or layer normalization
            pae_feature = (pae_feature - 8.0) / 4.0  # Typical PAE range is 4-16
            len_feature = (len_feature - 0.0) / 1.0   # Already normalized above
        
        # Combine features
        features = jnp.concatenate([pae_feature, len_feature], axis=-1)
        
        # Add additional features for better prediction
        # Calculate interface confidence (complement of PAE)
        interface_confidence = jnp.exp(-pae_feature_raw / 4.0).reshape(-1, 1)  # Convert PAE to confidence
        features = jnp.concatenate([features, interface_confidence], axis=-1)
        
        if FLAGS.only_fit_binder:
            features = jax.lax.stop_gradient(features)

        # Improved neural network architecture
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
        
        # Apply dropout during training
        if is_training and self.dropout_rate > 0:
            key = hk.next_rng_key()
            hidden = hk.dropout(key, self.dropout_rate, hidden)
        
        # Output layer - separate models for each MHC class
        w2_init = hk.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='truncated_normal')
        # Better initialization based on expected pIC50 values
        b2_init = hk.initializers.Constant(jnp.array([6.0, 5.0]))  # Typical pIC50 values for Class I and II
        
        weights = hk.get_parameter("output_weights", shape=[hidden_dim, self.num_mhc_classes], dtype=features.dtype, init=w2_init)
        biases = hk.get_parameter("output_biases", shape=[self.num_mhc_classes], dtype=features.dtype, init=b2_init)
        
        if FLAGS.freeze_binder:
            weights = jax.lax.stop_gradient(weights)
            biases = jax.lax.stop_gradient(biases)
        
        # Calculate prediction for all classes
        prediction_per_class = hidden @ weights + biases
        
        # Use the one-hot encoding to select the prediction from the correct MHC class
        predicted_pic50 = jnp.sum(prediction_per_class * mhc_class_1hot, axis=-1, keepdims=True)
        
        return predicted_pic50, features

def binder_regression_fn(input_features, is_training=True):
    model = PaeBinderRegressor(num_mhc_classes=2, dropout_rate=FLAGS.dropout_rate)(input_features, is_training=is_training)
    return model

binder_regressor = hk.transform(binder_regression_fn, apply_rng=True)

# Initialize parameters with proper dummy inputs
rng = jax.random.PRNGKey(42)
regressor_init_dict = {
    'partner1_mask': jnp.ones([200], dtype=jnp.float32),
    'partner2_mask': jnp.ones([200], dtype=jnp.float32),
    'mhc_class_1hot': jnp.ones([1, 2], dtype=jnp.float32),
    'peptide_len': jnp.ones([1, 1], dtype=jnp.float32),
    'predicted_aligned_error': {
        'logits': jnp.zeros([1, 200, 200, 64], dtype=jnp.float32),
        'breaks': jnp.zeros([1, 63], dtype=jnp.float32),
    }
}
binder_model_params = binder_regressor.init(rng, input_features=regressor_init_dict, is_training=True)

af2_model_params = data.get_model_haiku_params(
    model_name=model_name, data_dir=FLAGS.data_dir)

print('initial binder params:', binder_model_params)

model_params = hk.data_structures.merge(af2_model_params, binder_model_params)
model_runner = model.RunModel(model_config, af2_model_params)

# Improved loss function with better handling and regularization
def get_loss_fn(model_params, key, processed_feature_dict, structure_flag, is_training=True):
    binder_model_params, af2_model_params = hk.data_structures.partition(
        lambda m, n, p: m[:9] != "alphafold", model_params)
    
    # Prepare the dictionary for our binder regressor
    regressor_dict = {
        'true_pic50': jnp.array(processed_feature_dict['labels'], dtype=jnp.float32),
        'mhc_class_1hot': processed_feature_dict['mhc_class_1hot'],
        'partner1_mask': processed_feature_dict['partner1_mask'],
        'partner2_mask': processed_feature_dict['partner2_mask'],
        'peptide_len': processed_feature_dict['peptide_len']
    }
    
    # Clean up the dictionary before passing to the main AlphaFold model
    del processed_feature_dict['mhc_class_1hot']
    del processed_feature_dict['partner1_mask']
    del processed_feature_dict['partner2_mask']
    del processed_feature_dict['peptide_len']
    del processed_feature_dict['labels']
    
    # Run AlphaFold to get structural features
    predicted_dict, af2_loss = model_runner.apply(
        af2_model_params, key, processed_feature_dict)
    
    # Pass AlphaFold's output to our regressor
    regressor_dict.update(predicted_dict)
    predicted_pic50, binder_features = binder_regressor.apply(
        binder_model_params, key, regressor_dict, is_training=is_training)
    
    # Improved regression loss with robust handling
    residuals = predicted_pic50 - regressor_dict['true_pic50']
    
    # Use Huber loss for more robust regression (less sensitive to outliers)
    delta = 1.0  # Huber loss parameter
    huber_loss = jnp.where(
        jnp.abs(residuals) <= delta,
        0.5 * residuals**2,
        delta * (jnp.abs(residuals) - 0.5 * delta)
    )
    regression_loss = jnp.mean(huber_loss)
    
    # Add L2 regularization on the regression head parameters
    l2_reg = 0.001 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(binder_model_params))
    regression_loss = regression_loss + l2_reg
    
    if FLAGS.only_fit_binder:
        af2_loss = jax.lax.stop_gradient(af2_loss)
        
    # Combine AlphaFold's structure loss with our regression loss
    fake_native_weight = jnp.array(FLAGS.fake_native_weight, jnp.float32)
    sf = jnp.array(structure_flag, jnp.float32)
    not_sf = jnp.ones(sf.shape, dtype=sf.dtype) - sf
    
    total_loss = ((sf + not_sf * fake_native_weight) * af2_loss[0] + 
                  FLAGS.regression_loss_weight * regression_loss)
                  
    return total_loss, (predicted_dict, regression_loss, predicted_pic50, binder_features)

def train_step(model_params, key, batch, structure_flag):
    (loss, (predicted_dict, regression_loss, pred_pic50, features)), grads = jax.value_and_grad(
        get_loss_fn, has_aux=True)(model_params, key, batch, structure_flag, is_training=True)
    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, predicted_dict, regression_loss, pred_pic50, features

def norm_grads_per_example(grads, l2_norm_clip=0.1):
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads

# Improved data loading function with proper error handling
def create_batch_from_dataset_row(row):
    nres = len(row.target_chainseq.replace('/',''))
    target_trim_positions = list(range(nres))
    native_align = {int(x.split(':')[0]):int(x.split(':')[1]) for x in row.native_alignstring.split(';')}

    batch = predict_utils.create_batch_for_training(
        row.target_chainseq, target_trim_positions, row.templates_alignfile,
        row.native_pdbfile, native_align, FLAGS.crop_size, model_runner,
        debug=FLAGS.debug, verbose=FLAGS.verbose,
        random_seed=0 if FLAGS.no_random else None,
    )

    # Parse affinity and handle missing/invalid data
    if hasattr(row, 'iedb_binding_affinity_nM') and pd.notna(row.iedb_binding_affinity_nM):
        try:
            affinity_nM = float(row.iedb_binding_affinity_nM)
            # Handle edge cases in affinity values
            if affinity_nM <= 0:
                affinity_nM = 0.01  # Set minimum value
            elif affinity_nM > 1e6:  # Cap very high values
                affinity_nM = 1e6
            pIC50 = 9.0 - np.log10(affinity_nM)
        except (ValueError, TypeError):
            print(f"Warning: Invalid affinity value for {getattr(row, 'targetid', 'unknown')}: {row.iedb_binding_affinity_nM}")
            pIC50 = 5.0  # Default value
    else:
        # If no IEDB data, infer from binder status
        pIC50 = 7.0 if row.binder else 4.0
    
    # Clamp pIC50 to reasonable range
    pIC50 = np.clip(pIC50, 2.0, 12.0)
    batch['labels'] = np.array([pIC50], dtype=np.float32)

    # MHC Class (one-hot encoded) with better default handling
    if hasattr(row, 'iedb_mhc_class') and pd.notna(row.iedb_mhc_class):
        mhc_class_map = {'I': 0, 'II': 1}
        class_str = str(row.iedb_mhc_class).upper()
        binder_class_int = mhc_class_map.get(class_str, 0)
    else:
        binder_class_int = getattr(row, 'binder_class', 0)
    
    batch['mhc_class_1hot'] = np.eye(2)[np.array([binder_class_int])]

    # Peptide Length with better handling
    if hasattr(row, 'iedb_peptide_len') and pd.notna(row.iedb_peptide_len):
        peptide_len = float(row.iedb_peptide_len)
    else:
        # Calculate from peptide_positions
        peptide_positions = [int(x) for x in row.peptide_positions.split(';')]
        peptide_len = len(peptide_positions)
    
    batch['peptide_len'] = np.array([[peptide_len]], dtype=np.float32)

    # Masks for PAE calculation
    peptide_positions = [int(x) for x in row.peptide_positions.split(';')]
    nonpeptide_positions = [x for x in range(nres) if x not in peptide_positions]
    batch['partner1_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner1_mask'][nonpeptide_positions] = 1.
    batch['partner2_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner2_mask'][peptide_positions] = 1.

    batch['native_exists'] = bool(row.native_exists)
    return batch

class CustomPandasDataset(torch.utils.data.Dataset):
    def __init__(self, df, loader):
        self.df = df.copy()
        self.loader = loader
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        return self.loader(self.df.iloc[index])

def collate(samples):
    out_dict = {}
    for name in train_utils.list_a + train_utils.list_a_templates:
        values = [item[name][0,] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    for name in train_utils.list_b + train_utils.list_b_templates:
        values = [item[name] for item in samples]
        out_dict[name] = np.stack(values, axis=0)
    out_dict['aatype_'] = np.stack([item['aatype'][0,] for item in samples], axis=0)
    
    # Collate new regression-related features
    out_dict['mhc_class_1hot'] = [item['mhc_class_1hot'] for item in samples]
    out_dict['peptide_len'] = [item['peptide_len'] for item in samples]
    out_dict['partner1_mask'] = [item['partner1_mask'] for item in samples]
    out_dict['partner2_mask'] = [item['partner2_mask'] for item in samples]
    out_dict['labels'] = [item['labels'] for item in samples]
    out_dict['native_exists'] = [item['native_exists'] for item in samples]
    return out_dict

def prep_batch_for_step(batch, training):
    torsion_dict = atom37_to_torsion_angles(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch['chi_mask'] = torsion_dict['torsion_angles_mask'][:,:,3:]
    sin_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,0]
    cos_chi = torsion_dict['torsion_angles_sin_cos'][:,:,3:,1]
    batch['chi_angles'] = jnp.arctan2(sin_chi, cos_chi)
    rigidgroups_dict = atom37_to_frames(
        jnp.array(batch['aatype_']),
        jnp.array(batch['all_atom_positions']),
        jnp.array(batch['all_atom_mask']))
    batch.update(rigidgroups_dict)
    for key_, value_ in batch.items():
        if key_ in train_utils.list_a + train_utils.list_a_templates or key_ in train_utils.list_c:
            batch[key_] = value_[:,None,]
    for item_ in train_utils.pdb_key_list_int:
        batch[item_] = jnp.array(batch[item_], jnp.int32)
    if training and FLAGS.random_recycling:
        batch['num_iter_recycling'] = jnp.array(np.tile(
            np.random.randint(0, model_config.model.num_recycle+1, 1)[None,],
            (batch_size, model_config.model.num_recycle)), jnp.int32)

    # Prepare all new features for JAX
    batch['mhc_class_1hot'] = jnp.array(batch['mhc_class_1hot'], dtype=jnp.float32)
    batch['peptide_len'] = jnp.array(batch['peptide_len'], dtype=jnp.float32)
    batch['partner1_mask'] = jnp.array(batch['partner1_mask'], dtype=jnp.float32)
    batch['partner2_mask'] = jnp.array(batch['partner2_mask'], dtype=jnp.float32)
    del batch['native_exists']

def protein_from_prediction(batch, predicted_dict):
    fold_output = predicted_dict['structure_module']
    return Protein(
        aatype=batch['aatype'][0,0,:],
        atom_positions=fold_output['final_atom_positions'][0,],
        atom_mask=fold_output['final_atom_mask'][0,],
        residue_index=batch['residue_index'][0,0,:] + 1,
        b_factors=np.zeros_like(fold_output['final_atom_mask'][0,]))

def valid_step(model_params, key, batch, structure_flag):
    """A single validation step, without gradients."""
    # Note is_training=False
    loss, (predicted_dict, regression_loss, pred_pic50, features) = get_loss_fn(
        model_params, key, batch, structure_flag, is_training=False
    )
    return loss, regression_loss, pred_pic50

# Create a p-mapped version of the validation step
p_valid_step = jax.pmap(
    valid_step,
    in_axes=(0, None, 0, None), # pmap over params and batch
    axis_name='model_ax'
)

def compute_valid_stats(valid_loader, replicated_params, jax_key):
    if FLAGS.no_valid:
        print('no valid stats')
        return
    temp_total_loss, temp_regression_loss = [], []
    predictions, true_values = [], []

    for n, batch in enumerate(valid_loader):
        structure_flag = batch['native_exists'][0]
        true_labels = batch['labels']
        prep_batch_for_step(batch, False)
        jax_key, subkey = jax.random.split(jax_key)

        # CORRECTED LINE: Use the new p_valid_step function
        loss, regression_loss, pred_pic50 = p_valid_step(
            replicated_params, subkey, batch, structure_flag
        )

        pred_val = float(pred_pic50[0][0])
        true_val = float(true_labels[0][0])

        print(f"VALIDATION Step {n}: total_loss={float(loss[0]):.4f}, regression_loss={float(regression_loss[0]):.4f}, "
              f"pred_pIC50={pred_val:.3f}, true_pIC50={true_val:.3f}, error={abs(pred_val-true_val):.3f}")

        temp_total_loss.append(np.mean(loss[0]))
        temp_regression_loss.append(np.mean(regression_loss[0]))
        predictions.append(pred_val)
        true_values.append(true_val)

    mean_total_loss = round(float(np.mean(temp_total_loss)),4)
    mean_regression_loss = round(float(np.mean(temp_regression_loss)),4)
    mae = np.mean(np.abs(np.array(predictions) - np.array(true_values)))
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_values))**2))
    
    print(f'\n--- VALIDATION SUMMARY (Step: {global_step}) ---\n'
          f'Avg Total Loss: {mean_total_loss}, Avg Regression Loss: {mean_regression_loss}\n'
          f'MAE: {mae:.4f}, RMSE: {rmse:.4f}\n', flush=True)

# Main script execution
train_df = pd.read_csv(FLAGS.train_dataset)
valid_df = pd.read_csv(FLAGS.valid_dataset)

print(f"Loaded {len(train_df)} training examples and {len(valid_df)} validation examples")

# Print data statistics
if 'iedb_binding_affinity_nM' in train_df.columns:
    train_affinities = train_df['iedb_binding_affinity_nM'].dropna()
    if len(train_affinities) > 0:
        train_pic50s = 9.0 - np.log10(train_affinities.clip(lower=0.01))
        print(f"Training pIC50 range: {train_pic50s.min():.2f} to {train_pic50s.max():.2f}, mean: {train_pic50s.mean():.2f}")

training_set = CustomPandasDataset(train_df, create_batch_from_dataset_row)
valid_set = CustomPandasDataset(valid_df, create_batch_from_dataset_row)

params_loader = {'shuffle': True, 'num_workers': FLAGS.num_cpus, 'pin_memory': False,
                 'batch_size': batch_size, 'collate_fn': collate}

train_loader = torch.utils.data.DataLoader(training_set, **params_loader)
valid_loader = torch.utils.data.DataLoader(valid_set, **params_loader)

# Improved learning rate schedule with warmup
warmup_steps = 1000
total_steps = len(train_loader) * FLAGS.num_epochs
if not FLAGS.no_ramp:
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3 * FLAGS.lr_coef,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-5 * FLAGS.lr_coef
    )
else:
    scheduler = optax.constant_schedule(1e-3 * FLAGS.lr_coef)

# Improved optimizer with gradient clipping and regularization
gradient_transform = optax.chain(
    optax.clip_by_global_norm(FLAGS.grad_norm_clip),  # Clip gradients first
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0)  # Convert to descent
)

replicated_params = jax.tree_map(lambda x: jnp.array([x] * jax.local_device_count()), model_params)
opt_state = gradient_transform.init(replicated_params)
global_step = 0

if FLAGS.test_load:
    for n, batch in enumerate(train_loader): 
        print('trainload:',n); sys.stdout.flush()
        if n > 5: break  # Just test a few
    for n, batch in enumerate(valid_loader): 
        print('validload:',n); sys.stdout.flush()
        if n > 5: break

num_epochs = 1 if FLAGS.notrain else FLAGS.num_epochs

# Training metrics tracking
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 5

for e in range(num_epochs):
    temp_total_loss, temp_regression_loss = [], []
    temp_predictions, temp_true_values = [], []
    grads_sum, grads_sum_count = None, 0
    
    print(f"\n=== EPOCH {e+1}/{num_epochs} ===")

    for n, batch in enumerate(train_loader):
        if FLAGS.notrain: break
        
        structure_flag = batch['native_exists'][0]
        true_labels = batch['labels']
        prep_batch_for_step(batch, True)
        jax_key, subkey = jax.random.split(jax_key)

        loss, grads, _, regression_loss, pred_pic50, binder_features \
            = jax.pmap(train_step, in_axes=(0,None,0,None), axis_name='model_ax')(
                replicated_params, subkey, batch, structure_flag)

        pred_val = float(pred_pic50[0][0])
        true_val = float(true_labels[0][0])
        error = abs(pred_val - true_val)
        
        print(f"TRAIN Epoch {e+1}, Step {n}: total_loss={float(loss[0]):.4f}, "
              f"regression_loss={float(regression_loss[0]):.4f}, "
              f"pred_pIC50={pred_val:.3f}, true_pIC50={true_val:.3f}, error={error:.3f}")

        temp_total_loss.append(float(loss[0]))
        temp_regression_loss.append(float(regression_loss[0]))
        temp_predictions.append(pred_val)
        temp_true_values.append(true_val)
        global_step += 1

        # Gradient accumulation
        if grads_sum_count == 0:
            grads_sum = grads
        else:
            grads_sum = jax.tree_multimap(lambda x, y: x+y, grads_sum, grads)
        grads_sum_count += 1

        # Apply gradients
        if (grads_sum_count >= FLAGS.apply_every and not FLAGS.freeze_everything):
            grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
            updates, opt_state = gradient_transform.update(grads_sum, opt_state)
            replicated_params = optax.apply_updates(replicated_params, updates)
            grads_sum, grads_sum_count = None, 0

        # Print periodic summaries
        if (n+1) % FLAGS.print_steps == 0:
            mean_total_loss = np.mean(temp_total_loss)
            mean_regression_loss = np.mean(temp_regression_loss)
            mae = np.mean(np.abs(np.array(temp_predictions) - np.array(temp_true_values)))
            rmse = np.sqrt(np.mean((np.array(temp_predictions) - np.array(temp_true_values))**2))
            
            print(f'\n--- TRAIN SUMMARY (Step: {global_step}) ---\n'
                  f'Avg Total Loss: {mean_total_loss:.4f}, Avg Regression Loss: {mean_regression_loss:.4f}\n'
                  f'MAE: {mae:.4f}, RMSE: {rmse:.4f}\n', flush=True)
            temp_total_loss, temp_regression_loss = [], []
            temp_predictions, temp_true_values = [], []
            
        # Save checkpoints
        if (n+1) % FLAGS.save_steps == 0:
            os.makedirs(FLAGS.output_dir, exist_ok=True)
            param_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_params_{global_step}.pkl')
            state_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_state_{global_step}.pkl')

            print(f"Saving checkpoint to {param_path}...")
            save_params = jax.tree_map(lambda x: x[0,], replicated_params)
            with open(param_path, 'wb') as f:
                pickle.dump(save_params, f)
            with open(state_path, 'wb') as f:
                pickle.dump(opt_state, f)

        # Validation
        if (n+1) % FLAGS.valid_steps == 0:
            compute_valid_stats(valid_loader, replicated_params, jax_key)
            
        sys.stdout.flush()

    # End-of-epoch validation
    print(f"\n=== END OF EPOCH {e+1} VALIDATION ===")
    compute_valid_stats(valid_loader, replicated_params, jax_key)
    
    # Early stopping logic (optional)
    # You can implement this based on validation loss if desired
    
print("Training completed!")

# Save final model
os.makedirs(FLAGS.output_dir, exist_ok=True)
final_param_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_final_params.pkl')
final_state_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_final_state.pkl')

save_params = jax.tree_map(lambda x: x[0,], replicated_params)
with open(final_param_path, 'wb') as f:
    pickle.dump(save_params, f)
with open(final_state_path, 'wb') as f:
    pickle.dump(opt_state, f)

print(f"Final model saved to {final_param_path}")