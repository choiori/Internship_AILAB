from .resnet import ResNetFeature
from .lstm import PlainLSTM
import torch.nn as nn
import torch.nn.functional as F

class IITNet(nn.Module):
    def __init__(self, config, seq_len, device):
        super(IITNet, self).__init__()
        
        self.device = device
        self.config = config
        self.seq_len = seq_len
        self.resnet_config = config['model']['resnet']
        self.lstm_config = config['model']['lstm']
        self.feature = ResNetFeature(self.resnet_config, seq_len).to(self.device)
        self.lstm = PlainLSTM(self.lstm_config, 128, self.device).to(self.device)
        self.fc = nn.Linear(256, self.config['model']['fc']['num_classes']).to(self.device)
    
    def forward(self, x):
        out = self.feature(x) 
        out = self.lstm(out)        
        out = self.fc(out)
        
        return out
    
# def loss_fn_kd(student_outputs, teacher_outputs, labels, alpha, T):
#     KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1),
#                              F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + F.cross_entropy(student_outputs, labels) * (1. - alpha)
    
#     return KD_loss
    
    
     
 
    
        




        