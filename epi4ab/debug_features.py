#!/usr/bin/env python3
"""
Debug script to check feature dimensions
"""
import sys
sys.path.append('/home3/rudrapra001/FYP-Research/interface_prediction')

from run_setup.logging import ModelLogging
from run_setup.arguments import initiate_argument

# Parse arguments like the main script
args = initiate_argument()
logging = ModelLogging(args)

print("=== Feature Dimension Debug ===")
print(f"Use pretrained: {logging.use_pretrained}")
print(f"Pretrained model: {logging.pretrained_model}")
print(f"Pretrained dim: {logging.pretrained_dim}")
print(f"Use seq ff: {logging.use_seq_ff}")
print(f"Seq out: {getattr(logging, 'seq_out', 'Not calculated yet')}")
print(f"Use struct: {logging.use_struct}")
print(f"Feature columns number: {logging.feature_columns_number}")
print(f"Use antiberty: {logging.use_antiberty}")
print(f"Antiberty CDR list: {logging.antiberty_cdr_list}")
print(f"Antiberty max len dict: {logging.antiberty_max_len_dict}")
print(f"Antiberty max len: {logging.antiberty_max_len}")
print(f"Antiberty ff out: {logging.antiberty_ff_out}")
print(f"Use MHA on: {logging.use_mha_on}")

# Calculate expected dimensions
print("\n=== Expected Dimension Calculation ===")
total_dim = 0

if logging.use_pretrained:
    if logging.use_seq_ff:
        seq_dim = logging.seq_ff_out
    else:
        seq_dim = logging.pretrained_dim
    total_dim += seq_dim
    print(f"Sequence features: {seq_dim}")

if logging.use_struct:
    total_dim += logging.feature_columns_number
    print(f"Structure features: {logging.feature_columns_number}")

if logging.use_antiberty:
    if logging.use_mha_on == 'no':
        antiberty_dim = logging.antiberty_max_len * logging.antiberty_ff_out
        total_dim += antiberty_dim
        print(f"Antiberty features: {antiberty_dim} (max_len: {logging.antiberty_max_len} × ff_out: {logging.antiberty_ff_out})")
    else:
        print(f"Antiberty with MHA: dimensions calculated differently")

print(f"\nTotal expected input features: {total_dim}")
print(f"Current logging.in_feature (before calculation): {logging.in_feature}")
