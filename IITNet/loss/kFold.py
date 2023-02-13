import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import DataParallel 

import numpy as np
import os

from torch.utils.data import DataLoader
from model import IITNet
# from model_simple import IITNet
from EarlyStopping import EarlyStopping as ES
from loader_ import EEGDataLoader
# from loader_simple import EEGDataLoader
from balanced_loss import Loss
import numpy as np
import torch
import torch.nn.functional as F



    
class KFold():
    def __init__(self, k_folds, seq_len, args_):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seq_len = seq_len
        self.resnet_config = {'num_layers' :  50 ,  'seq_len': self.seq_len, 'dropout_rate': 0.5}
        self.lstm_config= {'num_classes' : 5 , 'input_size': 128 , 'hidden_size' : 128, 'num_layers' : 2}

  
        self.k_folds = k_folds
        self.batch_size = 256
        self.learning_rate = 0.005
        self.weight_decay = 0.000001
        self.train_max_epochs = 100

        
        self.mode = args_.mode
        self.loss_type = args_.loss_type
        
        gpu = args_.gpu.split(',')
        self.gpu = list(map(int, gpu))

        
    def KFoldIter(self):
        y_trues = np.zeros(0)
        y_preds = np.zeros((0, self.lstm_config['num_classes']))
        

        for i in range(1, self.k_folds+1):
            
            self.fold_num = i
            
            if self.fold_num == 1:  
                if self.mode == 'train':
                    print(f'[INFO] train mode')
                else:
                    print('[INFO] eval mode')
                    
                print(f'[INFO] loss type : {self.loss_type}')
                
            print(f'\n[INFO] {self.fold_num}번째 fold \n')
            
            # dataloader
            train_set = EEGDataLoader(self.seq_len, self.fold_num)
            train_num_per_class = train_set.samples_per_class
            print(f'[INFO] train_num_per_class: {train_num_per_class}')

            valid_set = EEGDataLoader(self.seq_len, self.fold_num, 'val')
            valid_num_per_class = valid_set.samples_per_class
            print(f'[INFO] valid_num_per_class: {valid_num_per_class}')
            
            test_set = EEGDataLoader(self.seq_len, self.fold_num, 'test')
            
            self.fold_num_per_class = [x+y for x,y in zip(train_num_per_class, valid_num_per_class)]
            print(f'[INFO] fold_num_per_class: {self.fold_num_per_class}')
            
            
            self.train_loader = DataLoader(train_set, batch_size = self.batch_size, shuffle = True)
            self.valid_loader = DataLoader(valid_set, batch_size = self.batch_size, shuffle = True)
            self.test_loader = DataLoader(test_set, batch_size = self.batch_size, shuffle = True)

            
            # fold
            y_true, y_pred = self.OneFold()

            # each fold results
            y_trues = np.concatenate([y_trues, y_true])
            y_preds = np.concatenate([y_preds, y_pred])
            
                        
        return y_trues, y_preds
    
    
    def OneFold(self):
        
        
        self.model = IITNet(self.resnet_config, self.lstm_config, self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids = self.gpu)
        self.model.cuda()
        
        
        # loss
        if self.loss_type == 'cb_focal_loss' or self.loss_type =='cb_cross_entropy':
            loss_tmp = self.loss_type[3:]
            self.criterion = Loss(loss_type = loss_tmp, samples_per_class = self.fold_num_per_class, class_balanced = True)
        elif self.loss_type == 'focal_loss':
            self.criterion = Loss(loss_type='focal_loss', samples_per_class = self.fold_num_per_class, class_balanced = False)
        elif self.loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()

            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)


        # mode
        if self.mode == 'train':
            self.train_valid()
        
        y_true, y_pred = self.evaluate()

        
        return y_true, y_pred





    def train_valid(self):
        
        early_stopping = ES(self.seq_len, self.fold_num, self.loss_type)

        
        for epoch in range(1, self.train_max_epochs + 1):
              
            self.model.train()
            
            correct, total, train_loss, batch_cnt = 0, 0, 0, 0
            
            
            for batch_idx, (X, Y) in enumerate(self.train_loader):
                
                    
                batch_cnt += 1
                
                total += Y.size(0)
                X, Y = X.to(self.device), Y.view(-1).to(self.device)
                print(Y.shape)
    
                
                outputs = self.model(X)
                print(outputs.shape)
                cost = self.criterion(outputs, Y)
                
                self.optimizer.zero_grad()
                cost.backward()
                self.optimizer.step()
                
                train_loss += cost.item()
                predicted = torch.argmax(outputs, 1)
                correct += predicted.eq(Y).sum().item()
                
                

            print(f"Train Epoch: {epoch},   Loss: {train_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")

        
            # validation
            self.model.eval()
            correct, total, valid_loss, batch_cnt = 0, 0, 0, 0
                        
            with torch.no_grad():
                for batch_idx, (X, Y) in enumerate(self.valid_loader):
                    
                                        
                    batch_cnt += 1
                    total += Y.size(0)
                    X, Y = X.to(self.device), Y.view(-1).to(self.device)
                    
                    outputs = self.model(X)
                    cost = self.criterion(outputs, Y)
                    
                    valid_loss += cost.item()
                    predicted = torch.argmax(outputs, 1)
                    correct += predicted.eq(Y).sum().item()
                    

                print(f"Valid Epoch: {epoch},   Loss: {valid_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")

            early_stopping(valid_loss, self.model)
            
            if early_stopping.early_stop:
                print('Early Stopping')
                break

 
    # test                            
    def evaluate(self):
                
        checkpoint = torch.load(f'./checkpoints/{self.loss_type}/seq{self.seq_len}/fold{self.fold_num}.pt')
        self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        correct, total, test_loss, batch_cnt = 0, 0, 0, 0
        
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.lstm_config['num_classes']))
        
        with torch.no_grad():
            
            for batch_idx, (X, Y) in enumerate(self.test_loader):
                
                
                batch_cnt += 1
                
                total += Y.size(0)
                X, Y = X.to(self.device), Y.view(-1).to(self.device)
                
                outputs = self.model(X)
                

                cost = self.criterion(outputs, Y)
                
                test_loss += cost.item()
                predicted = torch.argmax(outputs, 1)
                correct += predicted.eq(Y).sum().item()
                
                
                y_true = np.concatenate([y_true, Y.cpu().numpy()])
                y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])

            
            print(f"\nTest Loss: {test_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
            
            return y_true, y_pred
            