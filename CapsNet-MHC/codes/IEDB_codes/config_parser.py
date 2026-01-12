# -*- coding: utf-8 -*-

import os
import json

import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    def __init__(self, json_file):
        self.config = json.loads(open(json_file).read())
        # Use the device configured in the main script
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.cpu_device = torch.device("cpu")

    def _get(self, keys, default=None):
        """Safely fetch nested config values with a default."""
        cur = self.config
        for key in keys:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return default
        return cur


    @property
    def max_len_hla(self):
        return self._get(['Data', 'max_len_hla'], self.config.get('max_len_hla'))

    @property
    def max_len_pep(self):
        return self._get(['Data', 'max_len_pep'], self.config.get('max_len_pep'))

    @property
    def batch_size(self):
        return self._get(['Training', 'batch_size'], self.config.get('batch_size', 1))

    @property
    def working_dir(self):
        path = self._get(['Paths', 'working_dir'], self.config.get('working_dir', './results_capsnet'))
        return os.path.join(BASE_DIR, path)

    @property
    def data_file(self):
        data_path = self._get(['Data', 'data_file'])
        if data_path is not None:
            base = os.path.join(BASE_DIR, '../..', 'dataset/IEDB_dataset')
            return os.path.join(base, data_path)

        if 'data_file' in self.config:
            path = self.config['data_file']
            return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

        raise KeyError('data_file')

    @property
    def test_file(self):
        test_path = self._get(['Data', 'test_file'])
        if test_path is not None:
            base = os.path.join(BASE_DIR, '../..', 'dataset/IEDB_dataset')
            return os.path.join(base, test_path)

        if 'test_file' in self.config:
            path = self.config['test_file']
            return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

        raise KeyError('test_file')

    @property
    def hla_seq_file(self):
        # Allow nested or flat config; default to legacy IEDB_dataset file
        seq_path = self._get(['Data', 'hla_seq_file'], self.config.get('hla_seq_file'))
        if seq_path is None:
            base = os.path.join(BASE_DIR, '../..', 'dataset/IEDB_dataset')
            return os.path.join(base, 'mhc_i_protein_seq2.txt')

        return seq_path if os.path.isabs(seq_path) else os.path.join(BASE_DIR, seq_path)


    @property
    def model_config(self):
        return self._get(['Model'], self.config)
    
    @property
    def dropout(self):
        return self.config.get('dropout', self._get(['Training', 'dropout'], 0.5))
    
    @property
    def grad_clip(self):
        return self._get(['Training', 'grad_clip'], self.config.get('grad_clip', 5))

    @property
    def start_lr(self):
        return self._get(['Training', 'start_lr'], self.config.get('start_lr', self.config.get('lr')))

    @property
    def min_lr(self):
        return self._get(['Training', 'min_lr'], self.config.get('min_lr', 1e-6))

    @property
    def epochs(self):
        return self._get(['Training', 'epochs'], self.config.get('epochs', 1))

    @property
    def loss_delta(self):
        return self._get(['Training', 'loss_delta'], self.config.get('loss_delta', 0))

    @property
    def encoding_method(self):
        return self._get(['Model', 'encoding_method'], self.config.get('encoding_method', 'Blosum62'))

    @property
    def encoding_method2(self):
        return self._get(['Model', 'encoding_method2'], self.config.get('encoding_method2', self.encoding_method))    

    @property
    def do_train(self):
        return self.config.get('do_train', True)

    @property
    def do_test(self):
        return self.config.get('do_test', False)

    @property
    def model_count(self):
        return self.config.get('model_count', self._get(['Training', 'model_count'], 1))

    def model_save_path(self,value):
        return os.path.join(self.working_dir, 'best_model_{}.pytorch'.format(value))
    
    @property
    def base_model_count(self):
        return self.config.get('base_model_count', 1)

    def model_base_path(self,value):
        return os.path.join(self.working_dir, 'base_model_{}.pytorch'.format(value))
