import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel

import os
import numpy as np
from torch.utils.data import DataLoader
from loader import EEGDataLoader

from utils import ES, args_to_list, progress_bar, load_module, metric_result_fold, seed_worker
from tqdm import tqdm

from collections import defaultdict

class KFoldIter():
    def __init__(self, args, config):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        self.config = config
        
        self.task = self.config['task'] 
        self.model_type = self.config['model']['type'] # lstm, transformer option
        
        self.mode = self.args.mode
        
        if self.args.kd == True:
            self.teacher_seq_len = self.config['teacher']['seq_len']
            self.student_seq_len = self.config['student']['seq_len']
        elif self.args.teacher == True:
            self.seq_len = self.config['teacher']['seq_len']
        elif self.args.student == True:
            self.seq_len = self.config['student']['seq_len']    

        self.start_fold_num = int(self.args.start_fold_num)
        
        self.k_fold = 20
        self.train_max_epochs = 100
        
        self.save_dir = self.args.save_dir
    
    
    def run(self):
        
        y_trues = np.zeros(0)
        y_preds = np.zeros((0, self.config['model']['fc']['num_classes']))
        
        # Train Target : kd
        if self.args.kd is True:
            if self.args.specify_fold_num is not False:
                self.fold_num = int(self.args.specify_fold_num)
                print(f'[INFO] {self.mode} mode')
                y_true, y_pred = self.kd_one_fold()
                
                y_trues = np.concatenate([y_trues, y_true])
                y_preds = np.concatenate([y_preds, y_pred])
                
                metric_result_fold([self.teacher_seq_len, self.student_seq_len], y_trues, y_preds, self.task, self.model_type, self.fold_num, 'kd')
                        
                exit()
   
            else:
                for fold_num in range(self.start_fold_num, self.k_fold + 1):
                    self.fold_num = fold_num
                    if fold_num == 1 :
                        print(f'[INFO] {self.mode} mode')
                    y_true, y_pred = self.kd_one_fold()
                    
                    y_trues = np.concatenate([y_trues, y_true])
                    y_preds = np.concatenate([y_preds, y_pred])
                
            return y_trues, y_preds    
        
        
        # Train Target : teacher or student model
        else:     
            if self.args.specify_fold_num is not False:
                self.fold_num = int(self.args.specify_fold_num)
                print(f'[INFO] {self.mode} mode')
                y_true, y_pred = self.one_fold()
                
                y_trues = np.concatenate([y_trues, y_true])
                y_preds = np.concatenate([y_preds, y_pred])
                
                if self.args.teacher: target = 'teacher'
                else: target = 'student'
                
                metric_result_fold(self.seq_len, y_trues, y_preds, self.task, self.model_type, self.fold_num, target)
                        
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.config['params']['learning_rate'], weight_decay = self.config['params']['weight_decay'])

        if self.save_dir is not False:
            if self.args.teacher is True:
                self.ckpt_path = f'ckpt/{self.save_dir}/{self.model_type}/teacher/seq_len{self.seq_len}'
            else: self.ckpt_path = f'ckpt/{self.save_dir}/{self.model_type}/student/seq_len{self.seq_len}'
        else:
            if self.args.teacher is True:
                self.ckpt_path = f'ckpt/{self.model_type}/teacher/seq_len{self.seq_len}'
            else: self.ckpt_path = f'ckpt/{self.model_type}/student/seq_len{self.seq_len}'

        self.ckpt_file = os.path.join(self.ckpt_path, f'fold_{self.fold_num}.pt')

        print(f'\n[INFO] {self.fold_num}번째 fold \n')

        if self.mode == 'train':
            self.train_valid()
        
        y_true, y_pred = self.eval(self.model)
                
        return y_true, y_pred
        
    def train_valid(self):
        
        es = ES(self.ckpt_path, self.ckpt_file)
        for epoch_num in range(self.train_max_epochs):
            
            self.model.train()
            correct, total, train_loss, batch_cnt = 0, 0, 0, 0
                
            for batch_idx, (X, Y) in enumerate(self.dataloader_dict['train']):

                batch_cnt += 1
                total += Y.size(0)
                
                X, Y = X.to(self.device), Y.view(-1).to(self.device)
                outputs = self.model(X)

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
                for batch_idx, (X, Y) in enumerate(self.dataloader_dict['valid']):
                    
                    batch_cnt += 1

                    
                    total += Y.size(0)
                    
                    X, Y = X.to(self.device), Y.view(-1).to(self.device)
                    
                    outputs = self.model(X)
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
            
    def eval(self, model):
        checkpoint = torch.load(self.ckpt_file)
        model.load_state_dict(checkpoint, strict=False) #strict = False
        
        model.eval()
        self.criterion = nn.CrossEntropyLoss()
        correct, total, test_loss, batch_cnt = 0, 0, 0, 0
        
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.config['model']['fc']['num_classes']))
        
        with torch.no_grad():
            
            for batch_idx, (X, Y) in enumerate(self.dataloader_dict['test']):
                
                batch_cnt += 1

                total += Y.size(0)
                
                X, Y = X.to(self.device), Y.view(-1).to(self.device)
                
                outputs = model(X)
                loss = self.criterion(outputs, Y)
                
                
                test_loss += loss.item() # tensor value -> python number 
                predicted = torch.argmax(outputs, 1)
                correct += predicted.eq(Y).sum().item()
                            
                y_true = np.concatenate([y_true, Y.cpu().numpy()])
                y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])
                
                progress_bar(batch_idx, len(self.dataloader_dict['test']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


            print(f"Test Epoch: Loss: {test_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
            
            return y_true, y_pred


    def kd_one_fold(self):
        
        self.teacher_model, self.student_model = self.kd_build_model()
        self.student_model = torch.nn.DataParallel(self.student_model, device_ids = args_to_list(self.args.gpu))
        self.teacher_model = torch.nn.DataParallel(self.teacher_model, device_ids = args_to_list(self.args.gpu))
        self.student_model.cuda()
        self.teacher_model.cuda()
        

        self.dataloader_dict = self.kd_build_dataloader(self.fold_num)
        
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr = self.config['params']['learning_rate'], weight_decay = self.config['params']['weight_decay'])
        
        if self.save_dir is not False:
            self.teacher_ckpt_file = f'./ckpt/{self.save_dir}/{self.model_type}/teacher/seq_len{self.teacher_seq_len}/fold_{self.fold_num}.pt'
            self.student_ckpt_path = f'ckpt/{self.save_dir}/{self.model_type}/kd_{self.student_seq_len}/seq_len{self.student_seq_len}'
        else: 
            self.teacher_ckpt_file = f'./ckpt/{self.model_type}/teacher/seq_len{self.teacher_seq_len}/fold_{self.fold_num}.pt'
            self.student_ckpt_path = f'./ckpt/{self.model_type}/kd_{self.student_seq_len}/seq_len{self.student_seq_len}'
        
        self.student_ckpt_file = os.path.join(self.student_ckpt_path, f'fold_{self.fold_num}.pt')
        checkpoint = torch.load(self.teacher_ckpt_file, map_location=lambda storage, loc: storage)
        self.teacher_model.load_state_dict(checkpoint, strict=False)
        
        
        print(f'\n[INFO] {self.fold_num}번째 fold \n')

        if self.mode == 'train':
            self.kd_train_valid()
        
        y_true, y_pred = self.kd_eval(self.student_model)
                
        return y_true, y_pred



    def kd_train_valid(self):
            
            es = ES(self.student_ckpt_path, self.student_ckpt_file)
            for epoch_num in range(self.train_max_epochs):
                
                self.student_model.train()
                self.teacher_model.eval()
                correct, total, train_loss, batch_cnt = 0, 0, 0, 0
                    
                for batch_idx, (X, Y) in enumerate(self.dataloader_dict['train']):

                    batch_cnt += 1
                    total += Y.size(0)
                    
                    X, Y = X.to(self.device), Y.view(-1).to(self.device)
                    slice_idx = self.teacher_seq_len - self.student_seq_len
                    student_X = X[:, slice_idx:, :]
                    
                    student_outputs = self.student_model(student_X)
                    
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(X)
                        
                    loss = self.kd_loss_fn(student_outputs, teacher_outputs, Y )

                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item() # tensor value -> python number 
                    predicted = torch.argmax(student_outputs, 1)
                    correct += predicted.eq(Y).sum().item()
                    
                    progress_bar(batch_idx, len(self.dataloader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                    
                
                print(f"Train Epoch: {epoch_num},   Loss: {train_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
            
                self.student_model.eval()
        
                correct, total, valid_loss, batch_cnt = 0, 0, 0, 0
                
                with torch.no_grad():
                    for batch_idx, (X, Y) in enumerate(self.dataloader_dict['valid']):
                        
                        batch_cnt += 1

                        
                        total += Y.size(0)
                        
                        X, Y = X.to(self.device), Y.view(-1).to(self.device)
                        slice_idx = self.teacher_seq_len - self.student_seq_len
                        student_X = X[:, slice_idx:, :]
                        student_outputs = self.student_model(student_X)
                        loss = nn.CrossEntropyLoss()(student_outputs, Y)
                        valid_loss += loss.item() # tensor value -> python number 
                        predicted = torch.argmax(student_outputs, 1)
                        correct += predicted.eq(Y).sum().item()
                        
                        progress_bar(batch_idx, len(self.dataloader_dict['valid']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (valid_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                                
                    print(f"Valid Epoch: {epoch_num},   Loss: {valid_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")

                es(valid_loss, self.student_model)
                if es.early_stop:
                    print('[INFO] Early Stopping')
                    break
                
    def kd_eval(self, model):
        checkpoint = torch.load(self.student_ckpt_file)
        model.load_state_dict(checkpoint)
        
        model.eval()
        self.criterion = nn.CrossEntropyLoss()
        correct, total, test_loss, batch_cnt = 0, 0, 0, 0
        
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.config['model']['fc']['num_classes']))
        
        with torch.no_grad():
            
            for batch_idx, (X, Y) in enumerate(self.dataloader_dict['test']):
                
                batch_cnt += 1

                total += Y.size(0)
                
                X, Y = X.to(self.device), Y.view(-1).to(self.device)
                
                slice_idx = self.teacher_seq_len - self.student_seq_len
                student_X = X[:, slice_idx:, :]

                outputs = model(student_X)
                loss = self.criterion(outputs, Y)
                
                
                test_loss += loss.item() # tensor value -> python number 
                predicted = torch.argmax(outputs, 1)
                correct += predicted.eq(Y).sum().item()
                            
                y_true = np.concatenate([y_true, Y.cpu().numpy()])
                y_pred = np.concatenate([y_pred, outputs.cpu().numpy()])
                
                progress_bar(batch_idx, len(self.dataloader_dict['test']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


            print(f"Test Epoch: Loss: {test_loss / (batch_cnt):.3f} | Acc: {100. * correct / total:.3f} ({correct}/{total})")
            
            return y_true, y_pred
    
    def build_model(self):
        model_module = load_module(self.model_type)
        model = model_module.IITNet(self.config, self.seq_len, self.device)
        return model    
    
    def build_dataloader(self, fold):
        
        g = torch.Generator()
        g.manual_seed(0)

        
        train_dataset = EEGDataLoader(self.seq_len, fold)
        valid_dataset = EEGDataLoader(self.seq_len, fold, 'valid')
        test_dataset = EEGDataLoader(self.seq_len, fold, 'test')

        
        train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = self.config['params']['batch_size'], worker_init_fn=seed_worker, generator=g)
        valid_dataloader = DataLoader(valid_dataset, shuffle = True, batch_size = self.config['params']['batch_size'], worker_init_fn=seed_worker, generator=g)
        test_dataloader = DataLoader(test_dataset, shuffle = True, batch_size = self.config['params']['batch_size'], worker_init_fn=seed_worker, generator=g)
        
        dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
        
        return dataloader_dict
        
    def kd_build_model(self):
            
        model_module = load_module(self.model_type)
        teacher_model = model_module.IITNet(self.config, self.teacher_seq_len, self.device)        
        student_model = model_module.IITNet(self.config, self.student_seq_len, self.device)
        
        return teacher_model, student_model    
    
    def kd_build_dataloader(self, fold):
        
        g = torch.Generator()
        g.manual_seed(0)

        train_dataset = EEGDataLoader(self.teacher_seq_len, fold)
        valid_dataset = EEGDataLoader(self.teacher_seq_len, fold, 'valid')
        test_dataset = EEGDataLoader(self.teacher_seq_len, fold, 'test')

        
        train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = self.config['params']['batch_size'], worker_init_fn=seed_worker, generator=g)
        valid_dataloader = DataLoader(valid_dataset, shuffle = True, batch_size = self.config['params']['batch_size'], worker_init_fn=seed_worker, generator=g)
        test_dataloader = DataLoader(test_dataset, shuffle = True, batch_size = self.config['params']['batch_size'], worker_init_fn=seed_worker, generator=g)
        
        dataloader_dict = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
        
        return dataloader_dict
    
    def kd_loss_fn(self, student_outputs, teacher_outputs, labels, alpha=0.1, T=10):
        #0.7 T=20
        KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs/T, dim=1),
                                F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + F.cross_entropy(student_outputs, labels) * (1. - alpha)
        
        return KD_loss
    
        # student_loss = F.cross_entropy(input=student_outputs, target=labels)
        # distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(student_outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
        # total_loss =  alpha*student_loss + (1-alpha)*distillation_loss

        # return total_loss
        
            
    def loss_fn(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)
        
        
        
            
        
        
            
            