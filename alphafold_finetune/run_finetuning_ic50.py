######################################################################################88
FREDHUTCH_HACKS = False # silly stuff Phil added for running on Hutch servers

if FREDHUTCH_HACKS:
    import os
    # I think this has to happen before importing tensorflow?
    os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'

import os
from os import popen
import sys
import time
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
flags.DEFINE_string('data_dir', "./",
                    help='location of alphafold params; passed to '
                    'data.get_model_haiku_params; should contain params/ subfolder')

flags.DEFINE_bool('only_fit_binder', False, help='if True, dont fit alphafold params')
flags.DEFINE_bool('freeze_binder', False, help='if True, dont fit binder params')
flags.DEFINE_bool('freeze_everything', False, help='if True, dont fit anything')
flags.DEFINE_bool('no_ramp', False, help='if True, dont ramp')
flags.DEFINE_bool('no_valid', False, help='if True, dont compute valid stats')
flags.DEFINE_bool('quick_valid', False, help='if True, only validate on first 50 examples for speed')
flags.DEFINE_integer('quick_valid_size', 50, help='Number of examples to use for quick validation')
flags.DEFINE_bool('no_random', False, help='if True, dont randomize')
flags.DEFINE_bool('random_recycling', False, help='if True, set num_iter_recycling '
                  'randomly during training')

# Enhanced regression-specific flags
flags.DEFINE_float('regression_loss_weight', 15.0, help='Weight for the pIC50 regression loss - optimized')
flags.DEFINE_bool('normalize_features', True, help='Whether to normalize input features')
flags.DEFINE_float('dropout_rate', 0.15, help='Dropout rate for regularization - increased for better generalization')
flags.DEFINE_bool('use_attention', True, help='Whether to use attention mechanism in feature processing')
flags.DEFINE_integer('patience_limit', 10, help='Number of validation steps without improvement before early stopping')
flags.DEFINE_float('huber_delta', 1.0, help='Huber loss delta parameter for robust regression')
flags.DEFINE_float('ranking_loss_weight', 0.1, help='Weight for ranking loss to preserve relative ordering')
flags.DEFINE_float('confidence_loss_weight', 0.05, help='Weight for prediction confidence regularization')

# Checkpoint and resumption flags
flags.DEFINE_string('resume_from_checkpoint', None, help='Path to checkpoint directory to resume training from')
flags.DEFINE_bool('save_training_state', True, help='Whether to save comprehensive training state in checkpoints')

FLAGS = flags.FLAGS

# GPU Memory optimization settings
if 'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ:
    memory_fraction = float(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'])
    print(f"🖥️  GPU Memory Fraction set to: {memory_fraction:.2f}")

# Enable memory growth and optimization flags
os.environ.setdefault('TF_FORCE_UNIFIED_MEMORY', '1')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_force_compilation_parallelism=1')

assert FLAGS.msa_clusters >= 5 # since reduce_msa_clusters_by_max_templates is True for model_1/model_1_ptm

batch_size = FLAGS.batch_size
if batch_size > 1:
    print('⚠️  WARNING: batch_size > 1 has not been fully tested - use with caution')
    print('   Consider using batch_size=1 for stability')

print(f"🔧 Configuration:")
print(f"   Batch size: {batch_size}")
print(f"   MSA clusters: {FLAGS.msa_clusters}")
print(f"   Extra MSA: {FLAGS.extra_msa}")
print(f"   Crop size: {FLAGS.crop_size}")
print(f"   Regression loss weight: {FLAGS.regression_loss_weight}")
print(f"   Dropout rate: {FLAGS.dropout_rate}")
print(f"   Learning rate coefficient: {FLAGS.lr_coef}")
print(f"   Patience limit: {FLAGS.patience_limit}")
print("")

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

# Enhanced PaeBinderRegressor with advanced architecture and multi-modal features
class PaeBinderRegressor(hk.Module):
    """
    Advanced regressor that predicts pIC50 using comprehensive structural and sequence features.
    Features: multi-layer architecture, attention mechanism, feature fusion, and robust normalization.
    
    A100-optimized: Much larger capacity to fully utilize GPU memory.
    """
    def __init__(self, num_mhc_classes=2, dropout_rate=0.1, hidden_dims=[512, 256, 128, 64], use_attention=True, *args, **kwargs):
        super().__init__(name="PaeBinderRegressor")
        self.num_mhc_classes = num_mhc_classes
        self.dropout_rate = dropout_rate
        self.hidden_dims = hidden_dims  # Much deeper and wider for A100
        self.use_attention = use_attention

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

        # Enhanced interface analysis with multiple PAE metrics
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
        
        # Normalize all features more gently to preserve variance
        if FLAGS.normalize_features:
            # Much gentler normalization to preserve signal
            avg_interface_pae = avg_interface_pae / 20.0  # PAE range 0-20, mild normalization
            min_interface_pae = min_interface_pae / 20.0
            max_interface_pae = max_interface_pae / 20.0
            std_interface_pae = std_interface_pae / 20.0
            len_feature_raw = len_feature_raw / 3.0  # Mild length normalization
            partner1_plddt = partner1_plddt / 100.0  # pLDDT range 0-100
            partner2_plddt = partner2_plddt / 100.0
            avg_plddt = avg_plddt / 100.0
        
        # Combine all features into a comprehensive feature vector with safety checks
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
        
        # Safety check: replace any NaN or inf values with safe defaults
        all_features = jnp.where(jnp.isfinite(all_features), 
                               all_features, 
                               jnp.zeros_like(all_features))
        
        if FLAGS.only_fit_binder:
            all_features = jax.lax.stop_gradient(all_features)

        # Advanced multi-layer neural network with residual connections
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
            
            # Layer normalization - SKIP for last layer to preserve diversity!
            # Last layer normalization squashes output range
            is_last_layer = (i == len(self.hidden_dims) - 1)
            if not is_last_layer:
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f'ln_{i}')(x)
            
            # Advanced activation function (Swish/SiLU performs better than tanh)
            x = jax.nn.swish(x)
            
            # Residual connection if dimensions match
            if residual is not None:
                x = x + 0.1 * residual  # Scaled residual for stability
            
            # Dropout for regularization
            if is_training and self.dropout_rate > 0:
                key = hk.next_rng_key()
                x = hk.dropout(key, self.dropout_rate, x)
        
        # PURE ORDINAL CLASSIFICATION (4 bins)
        # Bin 0: pIC50 < 5.0  (very weak/non-binders)
        # Bin 1: 5.0 ≤ pIC50 < 7.0  (weak)
        # Bin 2: 7.0 ≤ pIC50 < 9.0  (moderate/strong)
        # Bin 3: pIC50 ≥ 9.0  (very strong)
        
        # Single ordinal classification head
        ordinal_w_init = hk.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        ordinal_logits = hk.Linear(4, w_init=ordinal_w_init, name='ordinal_classifier')(x)
        
        if FLAGS.freeze_binder:
            ordinal_logits = jax.lax.stop_gradient(ordinal_logits)
        
        # Convert ordinal predictions to continuous pIC50 using bin midpoints
        # Bin midpoints: [3.5, 6.0, 8.0, 10.0]
        bin_midpoints = jnp.array([3.5, 6.0, 8.0, 10.0])
        ordinal_probs = jax.nn.softmax(ordinal_logits)
        predicted_pic50 = jnp.sum(ordinal_probs * bin_midpoints, axis=-1, keepdims=True)
        
        # Sanity check for finite values
        predicted_pic50 = jnp.where(jnp.isfinite(predicted_pic50), 
                                   predicted_pic50, 
                                   jnp.array([[6.0]]))  # Default to middle range if NaN
        
        return predicted_pic50, ordinal_logits, all_features

def binder_regression_fn(input_features, is_training=True):
    model = PaeBinderRegressor(
        num_mhc_classes=2, 
        dropout_rate=FLAGS.dropout_rate,
        hidden_dims=[512, 256, 128, 64],  # A100-optimized: larger capacity
        use_attention=True
    )
    predicted_pic50, ordinal_logits, features = model(input_features, is_training=is_training)
    return predicted_pic50, ordinal_logits

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

# Enhanced loss function with multiple objectives and curriculum learning
def get_loss_fn(model_params, key, processed_feature_dict, structure_flag, is_training=True, epoch=0):
    binder_model_params, af2_model_params = hk.data_structures.partition(
        lambda m, n, p: m[:9] != "alphafold", model_params)
    
    # Prepare the dictionary for our binder regressor
    regressor_dict = {
        'true_pic50': jnp.array(processed_feature_dict['labels'], dtype=jnp.float32),
        'binary_label': jnp.array(processed_feature_dict['binder'], dtype=jnp.int32),  # 0 or 1
        'mhc_class_1hot': processed_feature_dict['mhc_class_1hot'],
        'partner1_mask': processed_feature_dict['partner1_mask'],
        'partner2_mask': processed_feature_dict['partner2_mask'],
        'peptide_len': processed_feature_dict['peptide_len']
    }
    
    # Convert pIC50 to ordinal class (4 bins)
    # Class 0: pIC50 < 5.0 (weak), Class 1: 5.0-7.0 (moderate)
    # Class 2: 7.0-9.0 (strong), Class 3: >= 9.0 (very strong)
    true_pic50 = regressor_dict['true_pic50'].ravel()
    ordinal_label = jnp.where(true_pic50 < 5.0, 0,
                    jnp.where(true_pic50 < 7.0, 1,
                    jnp.where(true_pic50 < 9.0, 2, 3)))
    regressor_dict['ordinal_label'] = ordinal_label
    
    # Clean up the dictionary before passing to the main AlphaFold model
    del processed_feature_dict['mhc_class_1hot']
    del processed_feature_dict['partner1_mask']
    del processed_feature_dict['partner2_mask']
    del processed_feature_dict['peptide_len']
    del processed_feature_dict['labels']
    del processed_feature_dict['binder']
    
    # Run AlphaFold to get structural features
    predicted_dict, af2_loss = model_runner.apply(
        af2_model_params, key, processed_feature_dict)
    
    # Add pLDDT to regressor input if available
    if 'plddt' in predicted_dict:
        regressor_dict['plddt'] = predicted_dict['plddt']
    
    # Pass AlphaFold's output to our regressor
    regressor_dict.update(predicted_dict)
    predicted_pic50, ordinal_logits = binder_regressor.apply(
        binder_model_params, key, regressor_dict, is_training=is_training)
    
    # PURE ORDINAL CLASSIFICATION LOSS
    # Simple cross-entropy loss for 4-way classification
    ordinal_label = regressor_dict['ordinal_label']
    ordinal_label_onehot = jax.nn.one_hot(ordinal_label.ravel(), num_classes=4)
    ordinal_loss = optax.softmax_cross_entropy(
        ordinal_logits.ravel(),
        ordinal_label_onehot
    )
    ordinal_loss = jnp.mean(ordinal_loss)
    
    # Calculate ordinal accuracy
    ordinal_pred = jnp.argmax(ordinal_logits, axis=-1).ravel()
    ordinal_acc = jnp.mean(ordinal_pred == ordinal_label.ravel())
    
    # Ensure loss is finite
    ordinal_loss = jnp.where(jnp.isfinite(ordinal_loss), ordinal_loss, jnp.array(1.0))
    
    # Total binding loss is just ordinal classification loss
    total_binding_loss = ordinal_loss
    
    # Light L2 regularization
    l2_reg = 0.00001 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(binder_model_params))
    l2_reg = jnp.where(jnp.isfinite(l2_reg), l2_reg, 0.0)
    total_binding_loss = total_binding_loss + l2_reg
    
    # For compatibility with monitoring code
    true_pic50 = regressor_dict['true_pic50']
    residuals = predicted_pic50.ravel() - true_pic50.ravel()
    binary_acc = jnp.array(0.0)  # Not used anymore
    ranking_loss = jnp.array(0.0)
    
    if FLAGS.only_fit_binder:
        af2_loss = jax.lax.stop_gradient(af2_loss)
        # When only training binder, binding tasks dominate
        structure_weight = 0.0
        binding_weight = 1.0
    else:
        # Adaptive loss weighting between structure and binding prediction
        structure_weight = FLAGS.fake_native_weight * (1 - structure_flag) + structure_flag
        binding_weight = FLAGS.regression_loss_weight
    
    # Total loss: structure + ordinal classification
    total_loss = structure_weight * af2_loss[0] + binding_weight * total_binding_loss
    
    # Additional metrics for monitoring (MAE/MSE based on continuous predictions from ordinal)
    mae = jnp.mean(jnp.abs(residuals))
    mse = jnp.mean(residuals**2)
    
    return total_loss, (predicted_dict, total_binding_loss, predicted_pic50, ordinal_logits, mae, mse, ranking_loss, ordinal_acc, binary_acc)

def train_step(model_params, key, batch, structure_flag, epoch=0):
    # Simplified training step with pure ordinal classification
    def loss_fn_with_epoch(params, key, batch, structure_flag):
        return get_loss_fn(params, key, batch, structure_flag, is_training=True, epoch=epoch)
    
    (loss, (predicted_dict, binding_loss, pred_pic50, ordinal_logits, mae, mse, ranking_loss, ordinal_acc, binary_acc)), grads = jax.value_and_grad(
        loss_fn_with_epoch, has_aux=True)(model_params, key, batch, structure_flag)
    
    # Apply gradient clipping per example for stability
    grads = norm_grads_per_example(grads, l2_norm_clip=FLAGS.grad_norm_clip)
    
    # Parallel mean across devices
    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    binding_loss = jax.lax.pmean(binding_loss, axis_name='model_ax')
    mae = jax.lax.pmean(mae, axis_name='model_ax')
    mse = jax.lax.pmean(mse, axis_name='model_ax')
    ranking_loss = jax.lax.pmean(ranking_loss, axis_name='model_ax')
    ordinal_acc = jax.lax.pmean(ordinal_acc, axis_name='model_ax')
    binary_acc = jax.lax.pmean(binary_acc, axis_name='model_ax')
    
    return loss, grads, predicted_dict, binding_loss, pred_pic50, ordinal_logits, mae, mse, ranking_loss, ordinal_acc, binary_acc

def norm_grads_per_example(grads, l2_norm_clip=1.0):
    """Enhanced gradient clipping with per-parameter normalization"""
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    
    # Compute per-parameter norms and global norm
    per_param_norms = [jnp.linalg.norm(g.ravel()) for g in nonempty_grads]
    total_grad_norm = jnp.linalg.norm(jnp.array(per_param_norms))
    
    # Apply gradient clipping
    clip_factor = jnp.minimum(1.0, l2_norm_clip / (total_grad_norm + 1e-8))
    normalized_nonempty_grads = [g * clip_factor for g in nonempty_grads]
    
    return jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

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
    
    # Ordinal label for multi-task learning (4 bins based on pIC50)
    # Bin 0: pIC50 < 5.0 (weak binding)
    # Bin 1: 5.0 <= pIC50 < 7.0 (moderate binding)
    # Bin 2: 7.0 <= pIC50 < 9.0 (strong binding)
    # Bin 3: pIC50 >= 9.0 (very strong binding)
    if pIC50 < 5.0:
        ordinal_bin = 0
    elif pIC50 < 7.0:
        ordinal_bin = 1
    elif pIC50 < 9.0:
        ordinal_bin = 2
    else:
        ordinal_bin = 3
    batch['ordinal_bin'] = np.array([ordinal_bin], dtype=np.int32)
    
    # Binary label for multi-task learning (0 = non-binder, 1 = binder)
    batch['binder'] = np.array([int(row.binder)], dtype=np.int32)

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
    out_dict['ordinal_bin'] = [item['ordinal_bin'] for item in samples]  # Ordinal labels for multi-task learning
    out_dict['binder'] = [item['binder'] for item in samples]  # Binary labels for multi-task learning
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
    batch['ordinal_bin'] = jnp.array(batch['ordinal_bin'], dtype=jnp.int32)  # Preserve for multi-task learning
    batch['binder'] = jnp.array(batch['binder'], dtype=jnp.int32)  # Preserve for multi-task learning
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
    """Simplified validation step with ordinal classification."""
    # Note is_training=False for validation
    loss, (predicted_dict, binding_loss, pred_pic50, ordinal_logits, mae, mse, ranking_loss, ordinal_acc, binary_acc) = get_loss_fn(
        model_params, key, batch, structure_flag, is_training=False, epoch=0
    )
    return loss, binding_loss, pred_pic50, ordinal_logits, mae, mse, ranking_loss, ordinal_acc, binary_acc

# Create enhanced p-mapped version of the validation step
p_valid_step = jax.pmap(
    valid_step,
    in_axes=(0, None, 0, None),  # pmap over params and batch
    axis_name='model_ax'
)

def compute_valid_stats(valid_loader, replicated_params, jax_key):
    """Enhanced validation with multi-task metrics."""
    if FLAGS.no_valid:
        print('no valid stats')
        return None, None, None, None
    
    temp_total_loss, temp_binding_loss = [], []
    predictions, true_values = [], []
    all_maes, all_mses, all_ranking_losses, all_ordinal_accs, all_binary_accs = [], [], [], [], []

    # Quick validation mode for debugging
    max_valid_steps = FLAGS.quick_valid_size if FLAGS.quick_valid else float('inf')
    
    for n, batch in enumerate(valid_loader):
        # Early exit for quick validation
        if n >= max_valid_steps:
            print(f"Quick validation: stopping at {n} examples (requested {FLAGS.quick_valid_size})")
            break
            
        structure_flag = batch['native_exists'][0]
        true_labels = batch['labels']
        true_binder = batch['binder']
        prep_batch_for_step(batch, False)
        jax_key, subkey = jax.random.split(jax_key)

        # Simplified validation step with ordinal classification
        loss, binding_loss, pred_pic50, ordinal_logits, mae, mse, ranking_loss, ordinal_acc, binary_acc = p_valid_step(
            replicated_params, subkey, batch, structure_flag
        )

        pred_val = float(pred_pic50[0][0])
        true_val = float(true_labels[0][0])
        ordinal_pred = int(jnp.argmax(ordinal_logits[0]))
        
        # Calculate true ordinal bin for display
        if true_val < 5.0:
            true_bin = 0
        elif true_val < 7.0:
            true_bin = 1
        elif true_val < 9.0:
            true_bin = 2
        else:
            true_bin = 3

        print(f"VALIDATION Step {n}: total_loss={float(loss[0]):.4f}, binding_loss={float(binding_loss[0]):.4f}, "
              f"pred_pIC50={pred_val:.3f}, true_pIC50={true_val:.3f}, error={abs(pred_val-true_val):.3f}, "
              f"ordinal_pred={ordinal_pred}, true_bin={true_bin}, "
              f"ordinal_acc={float(ordinal_acc[0]):.3f}, "
              f"mae={float(mae[0]):.4f}, rmse={float(jnp.sqrt(mse[0])):.4f}")

        temp_total_loss.append(float(loss[0]))
        temp_binding_loss.append(float(binding_loss[0]))
        predictions.append(pred_val)
        true_values.append(true_val)
        all_maes.append(float(mae[0]))
        all_mses.append(float(mse[0]))
        all_ranking_losses.append(float(ranking_loss[0]))
        all_ordinal_accs.append(float(ordinal_acc[0]))
        all_binary_accs.append(float(binary_acc[0]))

    # Comprehensive validation metrics
    mean_total_loss = float(np.mean(temp_total_loss))
    mean_binding_loss = float(np.mean(temp_binding_loss))
    overall_mae = np.mean(np.abs(np.array(predictions) - np.array(true_values)))
    overall_rmse = np.sqrt(np.mean((np.array(predictions) - np.array(true_values))**2))
    overall_ranking_loss = np.mean(all_ranking_losses)
    overall_ordinal_acc = np.mean(all_ordinal_accs)
    overall_binary_acc = np.mean(all_binary_accs)
    
    # Additional metrics for model evaluation
    pred_std = np.std(predictions)
    correlation = np.corrcoef(predictions, true_values)[0, 1] if len(predictions) > 1 else 0.0
    
    validation_mode = f"QUICK ({len(predictions)} examples)" if FLAGS.quick_valid else f"FULL ({len(predictions)} examples)"
    print(f'\n--- ORDINAL CLASSIFICATION VALIDATION SUMMARY ({validation_mode}, Step: {global_step}) ---\n'
          f'Total Loss: {mean_total_loss:.4f}, Binding Loss: {mean_binding_loss:.4f}\n'
          f'Ordinal Accuracy: {overall_ordinal_acc:.4f} (4 bins: <5, 5-7, 7-9, ≥9)\n'
          f'Continuous Predictions (from ordinal): MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}\n'
          f'Correlation: {correlation:.4f}, Prediction Std: {pred_std:.4f}\n'
          f'Prediction Range: [{min(predictions):.3f}, {max(predictions):.3f}]\n'
          f'True Range: [{min(true_values):.3f}, {max(true_values):.3f}]\n')
    
    return mean_total_loss, mean_binding_loss, overall_mae, overall_rmse

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

# Advanced learning rate schedule with multiple phases
warmup_steps = max(300, len(train_loader) // 3)  # Balanced warmup
total_steps = len(train_loader) * FLAGS.num_epochs

if not FLAGS.no_ramp:
    # Start with meaningful LR immediately - no ultra-slow warmup from 0
    # With lr_coef=2.0, we get: init=4e-4, peak=2e-3, end=2e-5
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.2 * 1e-3 * FLAGS.lr_coef,  # Start at 20% of peak (not zero!)
        peak_value=1e-3 * FLAGS.lr_coef,  # Peak LR
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=1e-5 * FLAGS.lr_coef
    )
else:
    scheduler = optax.constant_schedule(5e-4 * FLAGS.lr_coef)  # Conservative constant

# Optimized optimizer configuration - using AdamW for better performance
gradient_transform = optax.chain(
    # 1. Aggressive gradient clipping for stability
    optax.clip_by_global_norm(0.5),
    
    # 2. AdamW optimizer - better than Adam for fine-tuning tasks
    optax.adamw(
        learning_rate=scheduler,
        b1=0.9,           # First moment decay rate
        b2=0.999,         # Second moment decay rate  
        eps=1e-8,         # Small constant for numerical stability
        weight_decay=1e-4 # Regularization
    )
)

replicated_params = jax.tree_map(lambda x: jnp.array([x] * jax.local_device_count()), model_params)
opt_state = gradient_transform.init(replicated_params)
global_step = 0
start_epoch = 0

# Enhanced checkpoint resumption functionality
if FLAGS.resume_from_checkpoint:
    checkpoint_dir = FLAGS.resume_from_checkpoint
    print(f"🔄 Resuming training from checkpoint: {checkpoint_dir}")
    
    # Find the latest checkpoint files
    import glob
    param_files = glob.glob(os.path.join(checkpoint_dir, "*_params.pkl"))
    state_files = glob.glob(os.path.join(checkpoint_dir, "*_state.pkl"))
    
    if not param_files or not state_files:
        print(f"❌ No checkpoint files found in {checkpoint_dir}")
        print("Available files:", os.listdir(checkpoint_dir))
        sys.exit(1)
    
    # Get the latest checkpoint
    latest_param_file = max(param_files, key=os.path.getctime)
    latest_state_file = max(state_files, key=os.path.getctime)
    
    print(f"📂 Loading parameters from: {latest_param_file}")
    print(f"📂 Loading optimizer state from: {latest_state_file}")
    
    try:
        # Load checkpoint data
        with open(latest_param_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        if isinstance(checkpoint_data, dict) and 'params' in checkpoint_data:
            # New format with training state
            loaded_params = checkpoint_data['params']
            start_epoch = checkpoint_data.get('epoch', 0)
            global_step = checkpoint_data.get('global_step', 0)
            best_val_loss = checkpoint_data.get('best_val_loss', float('inf'))
            best_val_mae = checkpoint_data.get('best_val_mae', float('inf'))
            training_history = checkpoint_data.get('training_history', {
                'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'learning_rates': []
            })
            print(f"✅ Resuming from epoch {start_epoch}, step {global_step}")
            print(f"📊 Best validation MAE so far: {best_val_mae:.4f}")
        else:
            # Legacy format - just parameters
            loaded_params = checkpoint_data
            print("⚠️  Legacy checkpoint format - starting fresh metrics")
        
        # Load optimizer state
        with open(latest_state_file, 'rb') as f:
            opt_state = pickle.load(f)
        
        # Update model parameters
        replicated_params = jax.tree_map(lambda x: jnp.array([x] * jax.local_device_count()), loaded_params)
        print("✅ Successfully loaded checkpoint!")
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("Starting training from scratch...")
        start_epoch = 0
        global_step = 0

if FLAGS.test_load:
    for n, batch in enumerate(train_loader): 
        print('trainload:',n); sys.stdout.flush()
        if n > 5: break  # Just test a few
    for n, batch in enumerate(valid_loader): 
        print('validload:',n); sys.stdout.flush()
        if n > 5: break

num_epochs = 1 if FLAGS.notrain else FLAGS.num_epochs

# Enhanced training metrics tracking and early stopping
# Initialize defaults (may be overridden by checkpoint loading)
if 'best_val_loss' not in locals():
    best_val_loss = float('inf')
if 'best_val_mae' not in locals():
    best_val_mae = float('inf')
if 'training_history' not in locals():
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'learning_rates': []
    }

patience_counter = 0
patience_limit = FLAGS.patience_limit

print(f"\n{'='*60}")
print(f"STARTING ENHANCED TRAINING WITH {num_epochs} EPOCHS")
if FLAGS.resume_from_checkpoint:
    print(f"RESUMING FROM EPOCH {start_epoch + 1}, STEP {global_step}")
print(f"{'='*60}")

for e in range(start_epoch, num_epochs):
    epoch_start_time = time.time()
    temp_total_loss, temp_binding_loss = [], []
    temp_predictions, temp_true_values = [], []
    temp_maes, temp_mses, temp_ranking_losses = [], [], []
    grads_sum, grads_sum_count = None, 0
    
    print(f"\n🚀 EPOCH {e+1}/{num_epochs} - Multi-Task Training")
    print(f"{'='*50}")

    for n, batch in enumerate(train_loader):
        if FLAGS.notrain: break
        
        structure_flag = batch['native_exists'][0]
        true_labels = batch['labels']
        prep_batch_for_step(batch, True)
        jax_key, subkey = jax.random.split(jax_key)

        # Ordinal classification training step
        loss, grads, _, binding_loss, pred_pic50, ordinal_logits, mae, mse, ranking_loss, ordinal_acc, binary_acc \
            = jax.pmap(train_step, in_axes=(0,None,0,None,None), axis_name='model_ax')(
                replicated_params, subkey, batch, structure_flag, e)

        pred_val = float(pred_pic50[0][0])
        true_val = float(true_labels[0][0])
        ordinal_pred = int(jnp.argmax(ordinal_logits[0]))
        
        # Calculate true ordinal bin
        if true_val < 5.0:
            true_bin = 0
        elif true_val < 7.0:
            true_bin = 1
        elif true_val < 9.0:
            true_bin = 2
        else:
            true_bin = 3
        
        error = abs(pred_val - true_val)
        current_lr = scheduler(global_step) if callable(scheduler) else FLAGS.lr_coef * 1e-3
        
        # Simplified logging for ordinal classification
        if n % 10 == 0 or n < 5:  # Print every 10 steps or first 5 steps
            print(f"📊 TRAIN Epoch {e+1}, Step {n}: "
                  f"loss={float(loss[0]):.4f}, binding_loss={float(binding_loss[0]):.4f}, "
                  f"ordinal_acc={float(ordinal_acc[0]):.3f} (pred={ordinal_pred}, true={true_bin}), "
                  f"mae={float(mae[0]):.4f}, rmse={float(jnp.sqrt(mse[0])):.4f}, "
                  f"pred_pIC50={pred_val:.3f}, true_pIC50={true_val:.3f}, lr={current_lr:.2e}")

        # Collect comprehensive metrics
        temp_total_loss.append(float(loss[0]))
        temp_binding_loss.append(float(binding_loss[0]))
        temp_predictions.append(pred_val)
        temp_true_values.append(true_val)
        temp_maes.append(float(mae[0]))
        temp_mses.append(float(mse[0]))
        temp_ranking_losses.append(float(ranking_loss[0]))
        global_step += 1

        # Gradient accumulation
        if grads_sum_count == 0:
            grads_sum = grads
        else:
            grads_sum = jax.tree_multimap(lambda x, y: x+y, grads_sum, grads)
        grads_sum_count += 1

        # Apply gradients with numerical stability checks
        if (grads_sum_count >= FLAGS.apply_every and not FLAGS.freeze_everything):
            grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
            
            # Check for NaN gradients before applying
            grad_finite = jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads_sum)
            # Use tree_leaves instead of tree_reduce for JAX 0.2.22 compatibility
            all_grads_finite = jnp.all(jnp.array(jax.tree_leaves(grad_finite)))
            
            if all_grads_finite:
                # AdamW requires current parameters for weight decay
                updates, opt_state = gradient_transform.update(grads_sum, opt_state, replicated_params)
                replicated_params = optax.apply_updates(replicated_params, updates)
            else:
                print(f"⚠️  WARNING: NaN or inf gradients detected at step {global_step}. Skipping update.")
            
            grads_sum, grads_sum_count = None, 0

        # Print periodic summaries with multi-task metrics
        if (n+1) % FLAGS.print_steps == 0:
            mean_total_loss = np.mean(temp_total_loss)
            mean_binding_loss = np.mean(temp_binding_loss)
            mean_mae = np.mean(temp_maes)
            mean_rmse = np.sqrt(np.mean(temp_mses))
            mean_ranking_loss = np.mean(temp_ranking_losses)
            overall_mae = np.mean(np.abs(np.array(temp_predictions) - np.array(temp_true_values)))
            overall_rmse = np.sqrt(np.mean((np.array(temp_predictions) - np.array(temp_true_values))**2))
            
            print(f'\n📈 ORDINAL CLASSIFICATION TRAIN SUMMARY (Step: {global_step}) 📈')
            print(f'{"="*55}')
            print(f'Total Loss: {mean_total_loss:.4f} | Binding Loss: {mean_binding_loss:.4f}')
            print(f'MAE: {overall_mae:.4f} | RMSE: {overall_rmse:.4f} | Ranking Loss: {mean_ranking_loss:.4f}')
            print(f'Batch MAE: {mean_mae:.4f} | Batch RMSE: {mean_rmse:.4f}')
            print(f'Learning Rate: {current_lr:.2e}')
            print(f'{"="*55}\n')
            
            # Store training history
            training_history['train_loss'].append(mean_total_loss)
            training_history['train_mae'].append(overall_mae)
            training_history['learning_rates'].append(current_lr)
            
            # Reset temporary storage
            temp_total_loss, temp_binding_loss = [], []
            temp_predictions, temp_true_values = [], []
            temp_maes, temp_mses, temp_ranking_losses = [], [], []
            
        # Save checkpoints with enhanced naming
        if (n+1) % FLAGS.save_steps == 0:
            os.makedirs(FLAGS.output_dir, exist_ok=True)
            param_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_epoch{e+1}_step{global_step}_params.pkl')
            state_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_epoch{e+1}_step{global_step}_state.pkl')

            print(f"💾 Saving enhanced checkpoint to {param_path}...")
            save_params = jax.tree_map(lambda x: x[0,], replicated_params)
            checkpoint_data = {
                'params': save_params,
                'epoch': e+1,
                'global_step': global_step,
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'best_val_mae': best_val_mae
            }
            with open(param_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            with open(state_path, 'wb') as f:
                pickle.dump(opt_state, f)

        # Enhanced validation with early stopping
        if (n+1) % FLAGS.valid_steps == 0:
            val_loss, val_reg_loss, val_mae, val_rmse = compute_valid_stats(valid_loader, replicated_params, jax_key)
            
            if val_loss is not None:  # Check if validation was run
                training_history['val_loss'].append(val_loss)
                training_history['val_mae'].append(val_mae)
                
                # Early stopping logic
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"🌟 NEW BEST VALIDATION MAE: {val_mae:.4f} (Loss: {val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"⚠️  No improvement for {patience_counter} validation steps (Best MAE: {best_val_mae:.4f})")
                    
                if patience_counter >= patience_limit:
                    print(f"🛑 EARLY STOPPING: No improvement for {patience_limit} validation steps")
                    break
        
        # EARLY SANITY CHECK: Validate after 50 steps to catch collapse immediately
        if global_step == 50 and e == start_epoch:
            print(f"\n{'='*60}")
            print("🔍 EARLY SANITY CHECK (after 50 steps)")
            print(f"{'='*60}")
            
            # Quick check on training predictions (if we have any in recent batch)
            if len(temp_predictions) > 0:
                pred_range = max(temp_predictions) - min(temp_predictions)
                pred_std = np.std(temp_predictions)
                true_range = max(temp_true_values) - min(temp_true_values)
                
                print(f"Recent training predictions: [{min(temp_predictions):.3f}, {max(temp_predictions):.3f}] (range: {pred_range:.3f}, std: {pred_std:.3f})")
                print(f"Recent true values: [{min(temp_true_values):.3f}, {max(temp_true_values):.3f}] (range: {true_range:.3f})")
            
            # Run quick validation
            val_loss, val_reg_loss, val_mae, val_rmse = compute_valid_stats(valid_loader, replicated_params, jax_key)
            
            # Check for collapse based on validation stats printed above
            print(f"\n⚠️  Check validation summary above for:")
            print(f"   - Prediction Std should be > 0.5 (currently shows in summary)")
            print(f"   - Prediction Range should be > 2.0 (currently shows in summary)")
            print(f"   - If both are low, MODEL IS COLLAPSING!")
            
            if len(temp_predictions) > 0 and pred_std < 0.5:
                print(f"\n⚠️⚠️⚠️  WARNING: RECENT TRAINING COLLAPSE DETECTED! ⚠️⚠️⚠️")
                print(f"Prediction std dev ({pred_std:.3f}) is too low - model predicting near-constant values!")
                print(f"This usually means the loss function or learning rate needs adjustment.")
                print(f"Current prediction range: [{min(temp_predictions):.3f}, {max(temp_predictions):.3f}]")
                print(f"Expected range should be similar to true values: [{min(temp_true_values):.3f}, {max(temp_true_values):.3f}]")
                print(f"\nConsider stopping training if predictions don't improve in next 100 steps.\n")
            else:
                print(f"\n✅ Sanity check PASSED - model showing good variance (std: {pred_std:.3f})")
                print(f"Predictions span {pred_range:.3f} units, comparable to true range {true_range:.3f}\n")
            
            print(f"{'='*60}\n")
            
        sys.stdout.flush()

    # End-of-epoch comprehensive evaluation
    epoch_time = time.time() - epoch_start_time
    print(f"\n⏰ EPOCH {e+1} COMPLETED in {epoch_time:.2f} seconds")
    print(f"{'='*60}")
    
    # Final epoch validation
    val_loss, val_reg_loss, val_mae, val_rmse = compute_valid_stats(valid_loader, replicated_params, jax_key)
    
    if val_loss is not None:
        print(f"📊 EPOCH {e+1} FINAL METRICS:")
        print(f"   Validation Loss: {val_loss:.4f}")
        print(f"   Validation MAE: {val_mae:.4f}")
        print(f"   Validation RMSE: {val_rmse:.4f}")
        print(f"   Best MAE so far: {best_val_mae:.4f}")
        
        # Check for early stopping at epoch level
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"🛑 EARLY STOPPING: No improvement for {patience_counter} epochs")
            break
    
    print(f"{'='*60}\n")
    
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