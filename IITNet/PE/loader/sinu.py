#%%
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn

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
    
    def __len__(self):
        return len(self.epochs)
    
    
    def __getitem__(self, idx):
        n_sample = 30 * self.sr * self.seq_len
        #########################################################
        file_idx, sub_idx, seq_len, pe = self.epochs[idx]
        #########################################################
        inputs = self.inputs[file_idx][sub_idx:sub_idx+seq_len]
        # inputs = inputs.reshape(1, n_sample)                      ####### for문 x #####
        inputs = torch.from_numpy(inputs).float()
        labels = self.labels[file_idx][sub_idx:sub_idx+seq_len]
        labels = torch.from_numpy(labels).long()
        labels = labels[self.target_idx]
                        
                           ##########
        return inputs, labels, pe
                           ##########
    
    def split_dataset(self):
    
        file_idx = 0
        inputs, labels, epochs = [], [], []
        data_root = os.path.join(self.dataset_path, self.signal_type)
        data_fname_list = sorted(os.listdir(data_root))
        data_fname_dict = {'train': [], 'test': [], 'valid': []}
        split_idx_list = np.load('idx_{}.npy'.format(self.dataset), allow_pickle=True)

        assert len(split_idx_list) == self.n_splits
        
        for i in range(len(data_fname_list)):
            subject_idx = int(data_fname_list[i][3:5]) # 환자 번호
            if subject_idx == self.fold - 1:
                data_fname_dict['test'].append(data_fname_list[i])
            elif subject_idx in split_idx_list[self.fold - 1]: # idx file -> valid
                data_fname_dict['valid'].append(data_fname_list[i])
            else:
                data_fname_dict['train'].append(data_fname_list[i])
       
            
        for data_fname in data_fname_dict[self.mode]:
            npz_file = np.load(os.path.join(data_root, data_fname))
            inputs.append(npz_file['x'])
            labels.append(npz_file['y'])
            #############################################################
            pe = PositionalEncoding(128, len(npz_file['y']) * 47, 'cpu')  
            for i in range(len(npz_file['y']) - self.seq_len + 1):
                epochs.append([file_idx, i, self.seq_len, pe.encoding[i * 47 : (i+self.seq_len) * 47]])
            #############################################################    
            file_idx += 1
    
        return inputs, labels, epochs


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len, device):
        """
        sin, cos encoding 구현
        
        parameter
        - d_model : model의 차원
        - max_len : 최대 seaquence 길이
        - device : cuda or cpu
        """
        
        super(PositionalEncoding, self).__init__() # nn.Module 초기화
        
        # input matrix(자연어 처리에선 임베딩 벡터)와 같은 size의 tensor 생성
        # 즉, (max_len, d_model) size
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False # 인코딩의 그래디언트는 필요 없다. 
        
        # 위치 indexing용 벡터
        # pos는 max_len의 index를 의미한다.
        pos = torch.arange(0, max_len, device =device)
        # 1D : (max_len, ) size -> 2D : (max_len, 1) size -> word의 위치를 반영하기 위해
        
        pos = pos.float().unsqueeze(dim=1) # int64 -> float32 (없어도 되긴 함)
        
        # i는 d_model의 index를 의미한다. _2i : (d_model, ) size
        # 즉, embedding size가 512일 때, i = [0,512]
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        # (max_len, 1) / (d_model/2 ) -> (max_len, d_model/2)
        self.encoding[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        