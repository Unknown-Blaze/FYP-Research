import os
import math
import random

import torch

############################################################################
# Data provider
############################################################################


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class DataProvider:
    def __init__(self, sequence_encode_func, sequence_encode_func2, data_file, test_file, batch_size, max_len_hla=273, max_len_pep=37,
        model_count=5, shuffle=True, hla_seq_file=None):
        self.batch_size = batch_size
        self.data_file = data_file

        self.test_file = test_file
        self.sequence_encode_func = sequence_encode_func
        self.sequence_encode_func2= sequence_encode_func2
        self.shuffle = shuffle
        self.max_len_hla = max_len_hla
        self.max_len_pep = max_len_pep
        self.model_count=model_count
        self.hla_seq_file = hla_seq_file or os.path.join(BASE_DIR, '../..','dataset/IEDB_dataset',  'mhc_i_protein_seq2.txt')

        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0
        # cache
        self.pep_encode_dict = {}
        self.hla_encode_dict = {}
        self.hla_encode_dict2= {}
        self.pep_encode_dict2= {}

        self.hla_sequence = {}
        self.read_hla_sequences()

        self.samples = []  
        self.train_samples = []
        self.validation_samples = []
        self.read_training_data()
        self.split_train_and_val()

        self.weekly_samples = []
        self.read_weekly_data()


    def train_steps(self):
        return math.ceil(len(self.train_samples[0]) / self.batch_size)

    def val_steps(self):
        return math.ceil(len(self.validation_samples[0]) / self.batch_size)

    def test_steps(self):
        return math.ceil(len(self.weekly_samples) / self.batch_size)

    def read_hla_sequences(self):

        file_path = self.hla_seq_file
        with open(file_path, 'r') as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split(' ')
                seq=info[1]
                if(len(seq)>=self.max_len_hla):
                   seq=seq[:self.max_len_hla]
                # print(info)
                self.hla_sequence[info[0]] = seq


    def read_weekly_data(self):
        skipped_long_pep = 0
        skipped_missing_mhc = 0
        
        with open(self.test_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                # SOTA Strict format: UID, Allele, Dummy, Peptide, Dummy, IC50
                uid = info[0]
                alleles = info[1]
                peptide = info[3]
                org_pep=peptide[:]
                if len(peptide) > self.max_len_pep:
                    skipped_long_pep += 1
                    continue
                hla_a = alleles
                if hla_a not in self.hla_sequence :
                    skipped_missing_mhc += 1
                    if skipped_missing_mhc <= 5:  # Show first 5
                        print(f"DEBUG: Skipping {hla_a} - not in MHC sequences")
                    continue
                # Use simple UID from first column (for SOTA strict format)
                # print(uid)
                self.weekly_samples.append((hla_a,peptide, uid))
        
        print(f"DEBUG: Loaded {len(self.weekly_samples)} test samples")
        print(f"DEBUG: Skipped {skipped_long_pep} samples (peptide too long)")
        print(f"DEBUG: Skipped {skipped_missing_mhc} samples (MHC not found)")
        print(f"DEBUG: Total MHC sequences loaded: {len(self.hla_sequence)}")
            
    def read_training_data(self):

        with open(self.data_file) as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue

                info = line.strip('\n').split('\t')
                # print(info)

                # SOTA Strict format: UID, Allele, Dummy, Peptide, Dummy, pIC50
                hla_a = info[1]

                if hla_a not in self.hla_sequence :
                    continue

                peptide = info[3]
                if len(peptide) > self.max_len_pep:
                    continue

                # NORMALIZATION: 
                # Standard mapping: 1.0 = Strong Binder, 0.0 = Non-Binder
                # pIC50 usually caps at ~12-14. Let's divide by a constant scale factor.
                # Or use the standard NetMHC normalization: 1 - log(IC50)/log(50000)
                # But since we already have pIC50, let's just scale it.
                raw_pic50 = float(info[5])
                # Scale 0-10 -> 0-1 (Assuming max pIC50 is roughly 10-12)
                ic50 = raw_pic50 / 12.0  
                # Clamp to 0-1 to be safe
                ic50 = max(0.0, min(1.0, ic50))

                self.samples.append((hla_a, peptide, ic50))

        if self.shuffle:
            random.shuffle(self.samples)

        # exit()

    def split_train_and_val(self):

        vd_count=math.ceil(len(self.samples)/max(self.model_count,5))
        for i in range(max(self.model_count-1,4)):
            self.validation_samples.append(self.samples[i*vd_count:(i+1)*vd_count])
            temp_sample=self.samples[:]
            del(temp_sample[i*vd_count:(i+1)*vd_count])
            self.train_samples.append(temp_sample)


        self.validation_samples.append(self.samples[len(self.samples)-vd_count:])
        temp_sample=self.samples[:]
        del(temp_sample[len(self.samples)-vd_count:])
        self.train_samples.append(temp_sample)


    def batch_train(self,order):
        """A batch of training data
        """
        data = self.batch(self.batch_index_train, self.train_samples[order])
        self.batch_index_train += 1
        return data

    def batch_val(self,order):
        """A batch of validation data
        """
        data = self.batch(self.batch_index_val, self.validation_samples[order])
        self.batch_index_val += 1
        return data

    def batch_test(self):
        """A batch of test data
        """
        data = self.batch(self.batch_index_test, self.weekly_samples, testing=True)
        self.batch_index_test += 1
        return data

    def new_epoch(self):
        """New epoch. Reset batch index
        """
        self.batch_index_train = 0
        self.batch_index_val = 0
        self.batch_index_test = 0

    def batch(self, batch_index, sample_set, testing=False):
        """Get a batch of samples
        """
        hla_a_tensors = []
        hla_a_mask = []
        hla_a_tensors2 = []
        hla_a_mask2 = []        
        pep_tensors = []
        pep_mask = []
        pep_tensors2 = []
        pep_mask2 = []
        ic50_list = []

        # for testing
        uid_list = []
        #validation_call
        sample_prototype=[]


        def encode_sample(sample):
            hla_a_allele = sample[0]
            # hla_b_allele = sample[1]
            pep = sample[1]

            if not testing:
                ic50 = sample[2]
            else:
                uid = sample[2]

            if hla_a_allele not in self.hla_encode_dict:
                hla_a_tensor, mask = self.sequence_encode_func(self.hla_sequence[hla_a_allele], self.max_len_hla)
                self.hla_encode_dict[hla_a_allele] = (hla_a_tensor, mask)

            hla_a_tensors.append(self.hla_encode_dict[hla_a_allele][0])
            hla_a_mask.append(self.hla_encode_dict[hla_a_allele][1])

            if hla_a_allele not in self.hla_encode_dict2:
                hla_a_tensor2, mask2 = self.sequence_encode_func2(self.hla_sequence[hla_a_allele], self.max_len_hla)
                self.hla_encode_dict2[hla_a_allele] = (hla_a_tensor2, mask2)

            hla_a_tensors2.append(self.hla_encode_dict2[hla_a_allele][0])
            hla_a_mask2.append(self.hla_encode_dict2[hla_a_allele][1])

            if pep not in self.pep_encode_dict:
                pep_tensor, mask = self.sequence_encode_func(pep, self.max_len_pep)              
                self.pep_encode_dict[pep] = (pep_tensor, mask)
            pep_tensors.append(self.pep_encode_dict[pep][0])
            pep_mask.append(self.pep_encode_dict[pep][1])

            if pep not in self.pep_encode_dict2:
                pep_tensor2, mask2 = self.sequence_encode_func2(pep, self.max_len_pep)
                self.pep_encode_dict2[pep] = (pep_tensor2, mask2)
            pep_tensors2.append(self.pep_encode_dict2[pep][0])
            pep_mask2.append(self.pep_encode_dict2[pep][1])


            if not testing:
                ic50_list.append(ic50)
            else:
                uid_list.append(uid)

        start_i = batch_index * self.batch_size
        end_i = start_i + self.batch_size
        for sample in sample_set[start_i: end_i]:
           # doesn't matter if the end_i exceed the maximum index
           #validation_call
           sample_prototype.append(sample)
           encode_sample(sample)
           # print(self.hla_encode_dict)
           # exit()
        

        # print("xxx")
        # in case last batch does not have enough samples, random get from previous samples
        if len(hla_a_tensors) < self.batch_size:
            if len(sample_set) < self.batch_size:
                for _ in range(self.batch_size - len(hla_a_tensors)):
                    #validation_call
                    temp=random.choice(sample_set)
                    sample_prototype.append(temp)                    
                    encode_sample(temp)

            else:
                for i in random.sample(range(start_i), self.batch_size - len(hla_a_tensors)):
                    #validation_call
                    sample_prototype.append(sample_set[i])
                    encode_sample(sample_set[i])

        if not testing:
     	
            return (
                torch.stack(hla_a_tensors, dim=0),
                torch.stack(hla_a_mask, dim=0),
                torch.stack(hla_a_tensors2, dim=0),
                torch.stack(hla_a_mask2, dim=0),


                torch.stack(pep_tensors, dim=0),
                torch.stack(pep_mask, dim=0),
                torch.stack(pep_tensors2, dim=0),
                torch.stack(pep_mask2, dim=0),

                torch.tensor(ic50_list),
                #validation_call
                sample_prototype  

            )
        else:
            return (
                torch.stack(hla_a_tensors, dim=0),
                torch.stack(hla_a_mask, dim=0),
                torch.stack(hla_a_tensors2, dim=0),
                torch.stack(hla_a_mask2, dim=0),

                torch.stack(pep_tensors, dim=0),
                torch.stack(pep_mask, dim=0),
                torch.stack(pep_tensors2, dim=0),
                torch.stack(pep_mask2, dim=0),

                uid_list,
            )
