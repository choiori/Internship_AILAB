import torch
import torch.nn as nn

import argparse
import yaml

from kfold_baseline import KFoldIter_baseline
from kfold import KFoldIter 
from utils import args_to_list, metric_result, load_module, set_random_seed

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default = 'eval')
    parser.add_argument('--config', default = './config/config.yaml')
    parser.add_argument('--seq_len', type=str, default = '10')
    parser.add_argument('--start_fold_num', default = 1)
    parser.add_argument('--specify_fold_num', default = False)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default= '0, 1, 2')
    parser.add_argument('--save_dir', default=False)
    args = parser.parse_args()
    
    set_random_seed(args.seed, use_cuda=True)
    
    with open(args.config) as cfg_file:
        config = yaml.safe_load(cfg_file)
        
    for seq_len in args_to_list(args.seq_len):
        print(f'[INFO] sequence_length : {seq_len}')
        
        if config['TASK'] == 'baseline':
            kFold = KFoldIter_baseline(args, config, seq_len)
        else:
            kFold = KFoldIter(args, config, seq_len)
            
        y_trues, y_preds = kFold.run()
        
        metric_result(seq_len, y_trues, y_preds, config['TYPE'])
        
    