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
flags.DEFINE_string('train_dataset', None, help='tsv file with training dataset. See '
                    'README for format info')
flags.DEFINE_string('valid_dataset', None, help='tsv file with validation dataset')
flags.DEFINE_integer('crop_size', 190, help='Max size of training example; set this '
                     'as low as possible for memory and speed')
flags.DEFINE_integer('num_epochs', 2, help='number of epochs')
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
flags.DEFINE_integer('valid_steps', 100000,
                     help='calc loss on whole valid set every valid_steps')
flags.DEFINE_integer('num_cpus', 0, help='number of (extra?) cpus for loading')
flags.DEFINE_float('lr_coef', 0.025, help='learning rate coefficient')
flags.DEFINE_float('fake_native_weight', 0.25, help='weight to apply to the alphafold '
                   'loss when using a predicted structure as the native')
flags.DEFINE_float('struc_viol_weight', 1.0, help='structural violation weight')
flags.DEFINE_integer('msa_clusters', 5, help='number of msa cluster sequences')
flags.DEFINE_integer('extra_msa', 1, help='number of extra msa sequences')
flags.DEFINE_integer('num_evo_blocks', 48, help='number of evoformer blocks')
flags.DEFINE_float('grad_norm_clip', 10.0, help='value to clip gradient norms, '
                   'per update')
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

# --- CHANGE START ---
# The original script had flags for plddt_binder, pae_binder_slope, etc.
# These are for classification and are no longer needed.
# We add a single new flag to control the weight of our new regression loss.
flags.DEFINE_float('regression_loss_weight', 1.0, help='Weight for the pIC50 regression loss.')
# --- CHANGE END ---

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

# --- CHANGE START ---
# We are replacing the entire "Binder stuff" section.
# The old code had PlddtBinderClassifier and PaeBinderClassifier.
# We replace them with a single, more powerful PaeBinderRegressor.

class PaeBinderRegressor(hk.Module):
    """
    Predicts pIC50 using PAE, peptide length, and MHC class.
    This model is more powerful as it uses multiple features and context.
    """
    def __init__(self, num_mhc_classes=2, *args, **kwargs):
        super().__init__(name="PaeBinderRegressor")
        self.num_mhc_classes = num_mhc_classes

    def __call__(self, input_features):
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

        paes12 = jnp.sum((paes * mask1[..., None]) * mask2[..., None, :], keepdims=True)
        paes21 = jnp.sum((paes * mask2[..., None]) * mask1[..., None, :], keepdims=True)
        
        # Calculate the raw features
        pae_feature_raw = 0.1 * (paes12 + paes21) / (2 * jnp.sum(mask1) * jnp.sum(mask2))
        len_feature_raw = (peptide_len - 9.0) / 3.0

        # --- THE FINAL, ROBUST FIX IS HERE ---
        # Force both features into a (1, 1) shape before concatenation.
        # .reshape(-1, 1) robustly creates a 2D column vector from the feature.
        pae_feature = pae_feature_raw.reshape(-1, 1)
        len_feature = len_feature_raw.reshape(-1, 1)

        # Now, concatenation is guaranteed to work as both arrays are 2D.
        features = jnp.concatenate([pae_feature, len_feature], axis=-1)
        # --- END OF FIX ---

        if FLAGS.only_fit_binder:
            features = jax.lax.stop_gradient(features)

        # Define a separate linear model (weights and bias) for each MHC class.
        num_features = features.shape[-1]
        w_init = hk.initializers.RandomNormal(stddev=0.02)
        b_init = hk.initializers.Constant(jnp.array([5.5, 4.5])) 

        weights = hk.get_parameter("weights", shape=[num_features, self.num_mhc_classes], dtype=features.dtype, init=w_init)
        biases = hk.get_parameter("biases", shape=[self.num_mhc_classes], dtype=features.dtype, init=b_init)

        if FLAGS.freeze_binder:
            weights = jax.lax.stop_gradient(weights)
            biases = jax.lax.stop_gradient(biases)

        # Calculate prediction for all classes: y = (X @ W) + b
        prediction_per_class = features @ weights + biases
        
        # Use the one-hot encoding to select the prediction from the correct MHC class model
        predicted_pic50 = jnp.sum(prediction_per_class * mhc_class_1hot, axis=-1, keepdims=True)

        return predicted_pic50, features
def binder_regression_fn(input_features):
    # We specify num_mhc_classes=2 for MHC Class I and II
    model = PaeBinderRegressor(num_mhc_classes=2)(input_features)
    return model

binder_regressor = hk.transform(binder_regression_fn, apply_rng=True)

# Initialize the parameters for our new regressor module
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
binder_model_params = binder_regressor.init(rng, input_features=regressor_init_dict)
# --- CHANGE END ---

af2_model_params = data.get_model_haiku_params(
    model_name=model_name, data_dir=FLAGS.data_dir)

print('initial binder params:', binder_model_params)

model_params = hk.data_structures.merge(af2_model_params, binder_model_params)
model_runner = model.RunModel(model_config, af2_model_params)

# --- CHANGE START ---
# The loss function is now completely changed for regression.
# We replace softmax_cross_entropy with Mean Squared Error (MSE).
def get_loss_fn(model_params, key, processed_feature_dict, structure_flag):
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
        binder_model_params, key, regressor_dict)
    
    # Calculate regression loss (Mean Squared Error)
    regression_loss = jnp.mean(jnp.square(predicted_pic50 - regressor_dict['true_pic50']))
    
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
        get_loss_fn, has_aux=True)(model_params, key, batch, structure_flag)
    grads = jax.lax.pmean(grads, axis_name='model_ax')
    loss = jax.lax.pmean(loss, axis_name='model_ax')
    return loss, grads, predicted_dict, regression_loss, pred_pic50, features
# --- CHANGE END ---

def norm_grads_per_example(grads, l2_norm_clip=0.1):
    nonempty_grads, tree_def = jax.tree_util.tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    grads = jax.tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)
    return grads

# --- CHANGE START ---
# This function is heavily modified to load all the new information from your dataset.
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

    # 1. Regression Label (pIC50)
    affinity_nM = float(row.iedb_binding_affinity_nM)
    pIC50 = 9.0 - np.log10(affinity_nM if affinity_nM > 0 else 0.01)
    batch['labels'] = np.array([pIC50], dtype=np.float32)

    # 2. MHC Class (one-hot encoded)
    mhc_class_map = {'I': 0, 'II': 1}
    class_str = str(row.get('iedb_mhc_class', 'I')) # Default to Class I if missing
    binder_class_int = mhc_class_map.get(class_str, 0)
    batch['mhc_class_1hot'] = np.eye(2)[np.array([binder_class_int])]

    # 3. Peptide Length
    batch['peptide_len'] = np.array([[float(row.iedb_peptide_len)]], dtype=np.float32)

    # 4. Masks for PAE calculation
    peptide_positions = [int(x) for x in row.peptide_positions.split(';')]
    nonpeptide_positions = [x for x in range(nres) if x not in peptide_positions]
    batch['partner1_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner1_mask'][nonpeptide_positions] = 1.
    batch['partner2_mask'] = np.zeros((FLAGS.crop_size,))
    batch['partner2_mask'][peptide_positions] = 1.

    batch['native_exists'] = bool(row.native_exists)
    return batch
# --- CHANGE END ---

class CustomPandasDataset(torch.utils.data.Dataset):
    def __init__(self, df, loader):
        self.df = df.copy()
        self.loader = loader
    def __len__(self):
        return self.df.shape[0]
    def __getitem__(self, index):
        return self.loader(self.df.iloc[index])

# --- CHANGE START ---
# Updated collate function to handle the new features.
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
# --- CHANGE END ---

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

    # --- CHANGE START ---
    # Prepare all new features for JAX.
    batch['mhc_class_1hot'] = jnp.array(batch['mhc_class_1hot'], dtype=jnp.float32)
    batch['peptide_len'] = jnp.array(batch['peptide_len'], dtype=jnp.float32)
    batch['partner1_mask'] = jnp.array(batch['partner1_mask'], dtype=jnp.float32)
    batch['partner2_mask'] = jnp.array(batch['partner2_mask'], dtype=jnp.float32)
    del batch['native_exists']
    # --- CHANGE END ---

def protein_from_prediction(batch, predicted_dict):
    fold_output = predicted_dict['structure_module']
    return Protein(
        aatype=batch['aatype'][0,0,:],
        atom_positions=fold_output['final_atom_positions'][0,],
        atom_mask=fold_output['final_atom_mask'][0,],
        residue_index=batch['residue_index'][0,0,:] + 1,
        b_factors=np.zeros_like(fold_output['final_atom_mask'][0,]))

# --- The main training loop and validation functions are restored but updated for regression logging ---
def compute_valid_stats(valid_loader, replicated_params, jax_key):
    if FLAGS.no_valid:
        print('no valid stats')
        return
    temp_total_loss, temp_regression_loss = [], []
    for n, batch in enumerate(valid_loader):
        structure_flag = batch['native_exists'][0]
        true_labels = batch['labels']
        prep_batch_for_step(batch, False)
        jax_key, subkey = jax.random.split(jax_key)

        loss, _, _, regression_loss, pred_pic50, _ \
            = jax.pmap(train_step, in_axes=(0,None,0,None), axis_name='model_ax')(
                replicated_params, subkey, batch, structure_flag)
        
        print(f"VALIDATION Step {n}: total_loss={float(loss[0]):.4f}, regression_loss(MSE)={float(regression_loss[0]):.4f}, "
      f"pred_pIC50={float(pred_pic50[0][0]):.4f}, true_pIC50={float(true_labels[0][0]):.4f}")
        temp_total_loss.append(np.mean(loss[0]))
        temp_regression_loss.append(np.mean(regression_loss[0]))

    mean_total_loss = round(float(np.mean(temp_total_loss)),4)
    mean_regression_loss = round(float(np.mean(temp_regression_loss)),4)
    print(f'\n--- VALIDATION SUMMARY (Step: {global_step}) ---\n'
          f'Avg Total Loss: {mean_total_loss}, Avg Regression Loss (MSE): {mean_regression_loss}\n', flush=True)

# Main script execution
train_df = pd.read_csv(FLAGS.train_dataset)
valid_df = pd.read_csv(FLAGS.valid_dataset)

training_set  = CustomPandasDataset(train_df, create_batch_from_dataset_row)
valid_set = CustomPandasDataset(valid_df, create_batch_from_dataset_row)

params_loader = {'shuffle': True, 'num_workers': FLAGS.num_cpus, 'pin_memory': False,
                 'batch_size': batch_size, 'collate_fn': collate}

train_loader = torch.utils.data.DataLoader(training_set, **params_loader)
valid_loader = torch.utils.data.DataLoader(valid_set, **params_loader)

scheduler = optax.linear_schedule(0.0, 1e-3, 1000, 0) if not FLAGS.no_ramp else optax.linear_schedule(1e-3, 1e-3, 1000, 0)
gradient_transform = optax.chain(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),
                                 optax.scale_by_schedule(scheduler),
                                 optax.scale(-1.0*FLAGS.lr_coef))

replicated_params = jax.tree_map(lambda x: jnp.array([x] * jax.local_device_count()), model_params)
opt_state = gradient_transform.init(replicated_params)
global_step = 0

if FLAGS.test_load:
    for n, batch in enumerate(train_loader): print('trainload:',n); sys.stdout.flush()
    for n, batch in enumerate(valid_loader): print('validload:',n); sys.stdout.flush()

num_epochs = 1 if FLAGS.notrain else FLAGS.num_epochs

for e in range(num_epochs):
    temp_total_loss, temp_regression_loss = [], []
    grads_sum, grads_sum_count = None, 0

    for n, batch in enumerate(train_loader):
        if FLAGS.notrain: break
        
        structure_flag = batch['native_exists'][0]
        true_labels = batch['labels']
        prep_batch_for_step(batch, True)
        jax_key, subkey = jax.random.split(jax_key)

        loss, grads, _, regression_loss, pred_pic50, binder_features \
            = jax.pmap(train_step, in_axes=(0,None,0,None), axis_name='model_ax')(
                replicated_params, subkey, batch, structure_flag)

        print(f"TRAIN Epoch {e}, Step {n}: total_loss={float(loss[0]):.4f}, "
      f"regression_loss(MSE)={float(regression_loss[0]):.4f}, "
      f"pred_pIC50={float(pred_pic50[0][0]):.4f}, true_pIC50={float(true_labels[0][0]):.4f}, "
      f"features(pae,len)={binder_features[0][0]}")

        temp_total_loss.append(np.mean(loss[0]))
        temp_regression_loss.append(np.mean(regression_loss[0]))
        global_step += 1

        if grads_sum_count == 0: grads_sum = grads
        else: grads_sum = jax.tree_multimap(lambda x, y: x+y, grads_sum, grads)
        grads_sum_count += 1

        if (grads_sum_count >= FLAGS.apply_every and not FLAGS.freeze_everything):
            grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
            grads_sum = norm_grads_per_example(grads_sum, l2_norm_clip=FLAGS.grad_norm_clip)
            updates, opt_state = gradient_transform.update(grads_sum, opt_state)
            replicated_params = optax.apply_updates(replicated_params, updates)
            grads_sum, grads_sum_count = None, 0

        if (n+1) % FLAGS.print_steps == 0:
            mean_total_loss = round(float(np.mean(temp_total_loss)), 4)
            mean_regression_loss = round(float(np.mean(temp_regression_loss)), 4)
            print(f'\n--- TRAIN SUMMARY (Step: {global_step}) ---\n'
                  f'Avg Total Loss: {mean_total_loss}, Avg Regression Loss (MSE): {mean_regression_loss}\n', flush=True)
            temp_total_loss, temp_regression_loss = [], []
            
        # This is the new, corrected saving block
        if (n+1) % FLAGS.save_steps == 0:
            # Create the output directory if it doesn't exist
            os.makedirs(FLAGS.output_dir, exist_ok=True)

            # Use os.path.join to safely create the full file path
            param_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_params_{global_step}.pkl')
            state_path = os.path.join(FLAGS.output_dir, f'{FLAGS.outprefix}_state_{global_step}.pkl')

            print(f"Saving checkpoint to {param_path}...")

            # Save the model parameters and optimizer state
            save_params = jax.tree_map(lambda x: x[0,], replicated_params)
            with open(param_path, 'wb') as f:
                pickle.dump(save_params, f)
            with open(state_path, 'wb') as f:
                pickle.dump(opt_state, f)

        if (n+1)%FLAGS.valid_steps==0:
            compute_valid_stats(valid_loader, replicated_params, jax_key)
        sys.stdout.flush()

    compute_valid_stats(valid_loader, replicated_params, jax_key)