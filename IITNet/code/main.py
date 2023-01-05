import sys
sys.path.append('..')

from kFold import KFold
from metric import result_metric

import argparse


if __name__ == '__main__':
    
    # CUDA_VISIBLE_DEVICES = 0,1
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default = 'eval')
    parser.add_argument('--loss_type', default = 'cross_entropy')
    parser.add_argument('--seq_len', default = '10')
    parser.add_argument('--gpu', default = '0,1')
    args_ = parser.parse_args()
    
    seq_len = args_.seq_len.split(',')
    seq_len = list(map(int, seq_len))

    for i in seq_len:
        print(f'[INFO] sequence_length : {i}')
        
        EEG_Kfold = KFold(20, i, args_)
            
        y_trues, y_preds  = EEG_Kfold.KFoldIter()
        result_metric(i, y_trues, y_preds, args_.loss_type)
        
        