import os
import torch
import numpy as np
from torch.utils.data import Dataset
import itertools

class EEGDataLoader(Dataset):

    def __init__(self, seq_len, fold, mode='train'):

        self.mode = mode
        self.fold = fold

        self.sr = 100
        self.dataset = 'Sleep-EDF'
        self.seq_len = seq_len # 1 ~ N
        self.target_idx = -1 # last epoch
        self.signal_type = 'Fpz-Cz'
        self.n_splits = 20 # 20-fold cross validation

        self.root_dir = '../'
        self.dataset_path = os.path.join(self.root_dir, 'dataset', self.dataset)
        self.inputs, self.labels, self.epochs = self.split_dataset()
        self.samples_per_class = self.get_samples_per_class()
        
        
        
    def __len__(self):
        return len(self.epochs)



    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        file_idx, sub_idx, seq_len = self.epochs[idx]
        inputs = self.inputs[file_idx][sub_idx:sub_idx+seq_len]
        inputs = torch.from_numpy(inputs).float()
        labels = self.labels[file_idx][sub_idx:sub_idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
                
        return inputs, labels
    
    
    
    def split_dataset(self):

        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.signal_type)
        data_fname_list = sorted(os.listdir(data_root))
        data_fname_dict = {'train': [], 'test': [], 'val': []}
        split_idx_list = np.load('idx_{}.npy'.format(self.dataset), allow_pickle=True)

        assert len(split_idx_list) == self.n_splits
        
        for i in range(len(data_fname_list)):
            subject_idx = int(data_fname_list[i][3:5]) # 환자 번호
            if subject_idx == self.fold - 1:
                data_fname_dict['test'].append(data_fname_list[i])
            elif subject_idx in split_idx_list[self.fold - 1]: # idx file -> valid
                data_fname_dict['val'].append(data_fname_list[i])
            else:
                data_fname_dict['train'].append(data_fname_list[i])
       
            
        for data_fname in data_fname_dict[self.mode]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            for i in range(len(npz_file['y']) - self.seq_len + 1):
                epochs.append([file_idx, i, self.seq_len])
            file_idx += 1
        
        return inputs, labels, epochs
    
    
    
    def get_samples_per_class(self):
        
        last_labels = list(itertools.chain(*self.labels))
        last_labels = last_labels[self.seq_len-1:]
        last_labels = np.asarray(last_labels)
        
        samples_per_class = np.unique(last_labels, return_counts=True)[1]
        
        return samples_per_class
        
