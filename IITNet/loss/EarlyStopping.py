import torch
import numpy as np
import os

class EarlyStopping(): #patience = 10
    def __init__(self, seq_len, fold, loss_type, patience=10, verbose=True, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None    # 검증 데이터셋에 대한 오차 최적화 값
        self.early_stop = False
        self.val_loss_min = np.Inf # np.Inf는 넘파이에서 무한대를 표현
        self.delta = delta
        
        self.seq_len = seq_len
        self.fold = fold
        self.loss_type = loss_type

        self.verbose = verbose
    
    def __call__(self, val_loss, model): 
        # epoch만큼 학습이 반복되면서 best_loss가 갱신되고, best_loss에 진전이
        # 없으면 조기 종료한 후 모델을 저장함
        score = -val_loss
    
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(self.fold, val_loss, model)
        
        elif score < self.best_score + self.delta:
            # best_score + delta가 score 보다 크면??
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(self.fold, val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, fold, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        os.makedirs(f'./checkpoints/{self.loss_type}/seq{self.seq_len}', exist_ok=True)
        self.path =  f'./checkpoints/{self.loss_type}/seq{self.seq_len}/fold{fold}.pt'
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss