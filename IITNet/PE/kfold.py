import torch
import torch.nn as nn
from torch.nn import DataParallel

import os
import numpy as np
from torch.utils.data import DataLoader

from utils import ES, args_to_list, progress_bar, load_module, metric_result_fold, seed_worker
from tqdm import tqdm

from collections import defaultdict

class KFoldIter():
    def __init__(self, args, config, seq_len ):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.config = config
        
        self.type = self.config['TYPE']
        self.task = self.config['TASK']
        self.loader = self.config['LOADER']
        self.model_type = self.config['MODEL_TYPE']
        
        self.mode = self.args.mode
        self.seq_len = seq_len

        self.start_fold_num = int(self.args.start_fold_num)
        
        self.k_fold = 20
        self.train_max_epochs = 100
        
        self.save_dir = self.args.save_dir
    
    
    def run(self):
        
        y_trues = np.zeros(0)
        y_preds = np.zeros((0, self.config['MODEL']['FC']['NUM_CLASSES']))
        if self.args.specify_fold_num is not False:
            self.fold_num = int(self.args.specify_fold_num)
            print(f'[INFO] {self.mode} mode')
            y_true, y_pred = self.one_fold()
            
            y_trues = np.concatenate([y_trues, y_true])
            y_preds = np.concatenate([y_preds, y_pred])
            
            metric_result_fold(self.seq_len, y_trues, y_preds, self.type, self.fold_num)
                    
            exit()

            
        else:
            for fold_num in range(self.start_fold_num, self.k_fold + 1):
                self.fold_num = fold_num
                if fold_num == 1 :
                    print(f'[INFO] {self.mode} mode')
                y_true, y_pred = self.one_fold()
                
                y_trues = np.concatenate([y_trues, y_true])
                y_preds = np.concatenate([y_preds, y_pred])
            
        return y_trues, y_preds    
            
    def one_fold(self):
        
        self.model = self.build_model()
        self.model = torch.nn.DataParallel(self.model, device_ids = args_to_list(self.args.gpu))
        self.model.cuda()
        


        self.dataloader_dict = self.build_dataloader(self.fold_num)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config['PARAMETERS']['LEARNING_RATE'], weight_decay = self.config['PARAMETERS']['WEIGHT_DECAY'])

        if self.save_dir is not False:
            self.ckpt_path = f'ckpt/{self.save_dir}/{self.task}/{self.type}/seq_len{self.seq_len}'
        else: self.ckpt_path = f'ckpt/{self.task}/{self.type}/seq_len{self.seq_len}'
        self.ckpt_file = os.path.join(self.ckpt_path, f'fold_{self.fold_num}.pt')

        print(f'\n[INFO] {self.fold_num}번째 fold \n')

        if self.mode == 'train':
            self.train_valid()
        
        y_true, y_pred = self.eval()
                
        return y_true, y_pred
        
    def train_valid(self):
        
        es = ES(self.ckpt_path, self.ckpt_file)
        for epoch_num in range(self.train_max_epochs):
            
            self.model.train()
            correct, total, train_loss, batch_cnt = 0, 0, 0, 0

            for batch_idx, (X, Y, pos_idx) in enumerate(self.dataloader_dict['train']):

                batch_cnt += 1
                total += Y.size(0)
                
                X, Y, pos_idx = X.to(self.device), Y.view(-1).to(self.device), pos_idx.to(self.device)
                outputs = self.model(X, pos_idx)

                loss = self.criterion(outputs, Y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() # tensor value -> python number 
                predicted = torch.argmax(outputs, 1)
                correct += predicted.eq(Y).sum().item()
                
                progress_bar(batch_idx, len(self.dataloader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                
            
            print(f"Train Epoch: {epoch_num},   Loss: {train_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
        
            self.model.eval()
            correct, total, valid_loss, batch_cnt = 0, 0, 0, 0
            
            with torch.no_grad():
                for batch_idx, (X, Y, pos_idx) in enumerate(self.dataloader_dict['valid']):
                    
                    batch_cnt += 1

                    
                    total += Y.size(0)
                    
                    X, Y, pos_idx = X.to(self.device), Y.view(-1).to(self.device), pos_idx.to(self.device)
                    
                    outputs = self.model(X, pos_idx)
                    loss = self.criterion(outputs, Y)
                    
                    valid_loss += loss.item() # tensor value -> python number 
                    predicted = torch.argmax(outputs, 1)
                    correct += predicted.eq(Y).sum().item()
                    
                    progress_bar(batch_idx, len(self.dataloader_dict['valid']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                            
                print(f"Valid Epoch: {epoch_num},   Loss: {valid_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")

            es(valid_loss, self.model)
            if es.early_stop:
                print('[INFO] Early Stopping')
                break
            
    def eval(self):
        checkpoint = torch.load(self.ckpt_file)
        self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        correct, total, test_loss, batch_cnt = 0, 0, 0, 0
        
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.config['MODEL']['FC']['NUM_CLASSES']))
        
        with torch.no_grad():
            
            for batch_idx, (X, Y, pos_idx) in enumerate(self.dataloader_dict['test']):
                
                batch_cnt += 1

                total += Y.size(0)
                
                X, Y, pos_idx = X.to(self.device), Y.view(-1).to(self.device), pos_idx.to(self.device)
                
                outputs = self.model(X, pos_idx)
                loss = self.criterion(outputs, Y)
                
                
                test_loss += loss.item() # tensor value -> python number 
                predicted = torch.argmax(outputs, 1)
                correct += predicted.eq(Y).sum().item()
                            
                y_true = np.concatenate([y_true, Y.cpu().numpy()])
                y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])
                
                progress_bar(batch_idx, len(self.dataloader_dict['test']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


            acc = (100. * correct / total)
            print(f"Test Epoch: Loss: {test_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
            
            return y_true, y_pred
    
    def build_model(self):
        model_module = load_module(self.model_type)
        model = model_module.IITNet(self.config, self.seq_len, self.device)
        return model    
    
    def build_dataloader(self, fold):
        loader_module = load_module(self.loader)
        
        g = torch.Generator()
        g.manual_seed(0)

        
        train_dataset = loader_module.EEGDataLoader(self.seq_len, fold)
        valid_dataset = loader_module.EEGDataLoader(self.seq_len, fold, 'valid')
        test_dataset = loader_module.EEGDataLoader(self.seq_len, fold, 'test')

        
        train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = self.config['PARAMETERS']['BATCH_SIZE'], worker_init_fn=seed_worker, generator=g)
        valid_dataloader = DataLoader(valid_dataset, shuffle = True, batch_size = self.config['PARAMETERS']['BATCH_SIZE'], worker_init_fn=seed_worker, generator=g)
        test_dataloader = DataLoader(test_dataset, shuffle = True, batch_size = self.config['PARAMETERS']['BATCH_SIZE'], worker_init_fn=seed_worker, generator=g)
        
        dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
        
        return dataloader_dict
        
        
        
            
        
        
            
            