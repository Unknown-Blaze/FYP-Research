import sys
import datetime
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
from torch.optim import lr_scheduler
from torch import optim
import torch.nn as nn

# GPU Configuration - Set to use specific GPU and control memory growth
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Change this to use different GPU (0, 1, 2, etc.)

# Configure PyTorch to use specific GPU and control memory growth
if torch.cuda.is_available():
    # Set the device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Control memory growth to prevent overuse
    torch.cuda.empty_cache()
    
    # Optional: Set memory fraction (uncomment if needed)
    # torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU")

from config_parser import Config
from model import (
    Model,
    weight_initial,
    count_parameters,
)
from data_provider import DataProvider
from logger import (
    setup_logging,
    log_to_file,
)
from callbacks import (
    ModelCheckPointCallBack,
    EarlyStopCallBack,
)
from result_writer import (
    weeekly_result_writer,
    write_metrics_file,
)
from seq_encoding import ENCODING_METHOD_MAP


#############################################################################################
#
# Test
#
#############################################################################################

def batch_test(model, device, data, config):
    hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, uid_list = data
    pred_ic50 = model(hla_a.to(device), hla_mask.to(device), hla_a2.to(device), hla_mask2.to(device), pep.to(device), pep_mask.to(device), pep2.to(device), pep_mask2.to(device))
    # print(pred_ic50)
    return pred_ic50, uid_list

def calculate_metrics(y_true, y_pred):
    """Calculate R², PCC, SCC, RMSE, MAE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    r2 = r2_score(y_true, y_pred)
    pcc, pcc_pval = pearsonr(y_true, y_pred)
    scc, scc_pval = spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'r2': r2,
        'pcc': pcc,
        'pcc_pval': pcc_pval,
        'scc': scc,
        'scc_pval': scc_pval,
        'rmse': rmse,
        'mae': mae
    }

def test(config, data_provider):

    if not config.do_test:
        log_to_file('Skip testing', 'Not enabled testing')
        return

    device = config.device
    
    temp_list=[] 
    ground_truth_dict = {}  # Store ground truth pIC50 values by uid
    
    for p in range(config.base_model_count):
        for k in range(config.model_count):
            # load and prepare model
            state_dict = torch.load(config.model_save_path(p*config.model_count+k), weights_only=True)
            model = Model(config, config.dropout)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            temp_dict={}
            data_provider.new_epoch()
            for _ in range(data_provider.test_steps()):
                data = data_provider.batch_test()
                
                with torch.no_grad():
                    # Extract ground truth if first model (they're the same across models)
                    if p == 0 and k == 0:
                        # Unpack data tuple - format: (hla_a, hla_mask, hla_a2, hla_mask2, pep, pep_mask, pep2, pep_mask2, uid_list)
                        # For test batches, we might need to check if ground truth is available
                        # If test data includes IC50, it would be in position 8 or 9
                        if len(data) > 9:  # If IC50 is included in test data
                            ic50_values = data[8]
                            uid_list = data[9]
                            for i, uid in enumerate(uid_list):
                                ground_truth_dict[uid] = ic50_values[i].item()
                    
                    pred_ic50, uid_list= batch_test(model, device, data, config)
                    #print(pred_ic50)
                    for i, uid in enumerate(uid_list):
                        temp_dict[uid] = pred_ic50[i].item()
            temp_list.append(temp_dict)

    # average score of the emsemble model
    result_dict=temp_list[0]
    if config.model_count>1:
        for k in range(1,config.model_count):
            for j in result_dict.keys():
                result_dict[j]+=temp_list[k][j]

    if config.base_model_count>1:
        for p in range(1,config.base_model_count):
            for k in range(config.model_count):
                for j in result_dict.keys():
                    result_dict[j]+=temp_list[p*config.model_count+k][j]

    for j in result_dict.keys():
        result_dict[j]=result_dict[j]/(config.model_count*config.base_model_count)

    #print(result_dict)
    result_file = weeekly_result_writer(result_dict, config)
    log_to_file('Testing result file', result_file)

    # Calculate and log metrics if ground truth is available
    if ground_truth_dict:
        y_true = []
        y_pred = []
        for uid in sorted(ground_truth_dict.keys()):
            if uid in result_dict:
                y_true.append(ground_truth_dict[uid])
                y_pred.append(result_dict[uid])
        
        if len(y_true) > 0:
            metrics = calculate_metrics(y_true, y_pred)
            log_to_file('R² Score', f"{metrics['r2']:.4f}")
            log_to_file('Pearson Correlation (PCC)', f"{metrics['pcc']:.4f} (p-value: {metrics['pcc_pval']:.2e})")
            log_to_file('Spearman Correlation (SCC)', f"{metrics['scc']:.4f} (p-value: {metrics['scc_pval']:.2e})")
            log_to_file('RMSE', f"{metrics['rmse']:.4f}")
            log_to_file('MAE', f"{metrics['mae']:.4f}")
            
            # Save metrics to a separate file
            metrics_file = os.path.join(config.working_dir, f'detailed_metrics_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
            with open(metrics_file, 'w') as f:
                f.write("Regression Metrics (pIC50)\n")
                f.write("="*50 + "\n")
                f.write(f"R² Score: {metrics['r2']:.4f}\n")
                f.write(f"Pearson Correlation (PCC): {metrics['pcc']:.4f} (p-value: {metrics['pcc_pval']:.2e})\n")
                f.write(f"Spearman Correlation (SCC): {metrics['scc']:.4f} (p-value: {metrics['scc_pval']:.2e})\n")
                f.write(f"RMSE: {metrics['rmse']:.4f}\n")
                f.write(f"MAE: {metrics['mae']:.4f}\n")
                f.write(f"Number of test samples: {len(y_true)}\n")
            log_to_file('Detailed metrics file', metrics_file)
    else:
        log_to_file('Warning', 'No ground truth available for metrics calculation')

    metric_file = write_metrics_file(result_file, config)
    log_to_file('Testing metric result file', metric_file)


#############################################################################################
#
# Main
#
#############################################################################################

def main():
    # parse config
    config_file = sys.argv[1]
    config = Config(config_file)

    # setup logger
    setup_logging(config.working_dir)

    # encoding func
    encoding_func = ENCODING_METHOD_MAP[config.encoding_method]
    encoding_func2= ENCODING_METHOD_MAP[config.encoding_method2]
    log_to_file('Encoding method2', config.encoding_method2)

    data_provider=[]
    for p in range(config.base_model_count):
        temp_provider = DataProvider(
             encoding_func,
             encoding_func2,
             config.data_file,
             config.test_file,
             config.batch_size,
             max_len_hla=config.max_len_hla,
             max_len_pep=config.max_len_pep,
               model_count=config.model_count,
               hla_seq_file=config.hla_seq_file
        )
        data_provider.append(temp_provider)
 
    log_to_file('max_len_hla', data_provider[0].max_len_hla)
    log_to_file('max_len_pep', data_provider[0].max_len_pep)
    
    test(config, data_provider[0])


if __name__ == '__main__':
    main()
