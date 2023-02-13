import torch
import torch.nn as nn

import argparse
import yaml

from kfold import KFoldIter
from utils import args_to_list, metric_result, load_module, set_random_seed

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default = 'eval')
    parser.add_argument('--config', default = './config/config.yaml')
    parser.add_argument('--kd', action='store_true')
    parser.add_argument('--teacher', action='store_true')
    parser.add_argument('--student', action='store_true')
    parser.add_argument('--start_fold_num', default = 1)
    parser.add_argument('--specify_fold_num', default = False)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default= '0, 1, 2, 3')
    parser.add_argument('--save_dir', default=False)
    args = parser.parse_args()
    
    set_random_seed(args.seed, use_cuda=True)
    
    with open(args.config) as cfg_file:
        config = yaml.safe_load(cfg_file)
    if args.kd == True:    
        print(f"[INFO] sequence_length : teacher_{config['teacher']['seq_len']}, student_{config['student']['seq_len']}")
    elif args.teacher == True:
        print(f"[INFO] sequence_length : teacher_{config['teacher']['seq_len']}")
    elif args.student == True:
        print(f"[INFO] sequence_length : student_{config['student']['seq_len']}")
    else: raise('학습할 모델을 선택해주세요. teacher or student or kd')
    kFold = KFoldIter(args, config)
    y_trues, y_preds = kFold.run()
    
    if args.kd == True:
        metric_result([config['teacher']['seq_len'], config['student']['seq_len']], y_trues, y_preds, config['task'], config['model']['type'], 'kd')
    elif args.teacher == True:
        metric_result(config['teacher']['seq_len'], y_trues, y_preds, config['task'], config['model']['type'], 'teacher')
    elif args.student == True:
        metric_result(config['student']['seq_len'], y_trues, y_preds, config['task'], config['model']['type'], 'student')
