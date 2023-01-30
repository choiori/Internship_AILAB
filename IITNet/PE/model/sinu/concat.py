import torch.nn as nn
import torch
from ..lstm import PlainLSTM
from ..resnet import ResNetFeature
 
    
class IITNet(nn.Module):
    def __init__(self, config, seq_len, device):
        super(IITNet, self).__init__()
        
        self.device = device
        self.config = config
        self.seq_len = seq_len
        self.resnet_config = config['MODEL']['RESNET']
        self.lstm_config = config['MODEL']['LSTM']
        self.feature = ResNetFeature(self.resnet_config, seq_len).to(self.device)
        self.lstm = PlainLSTM(self.lstm_config, 256, self.device).to(self.device) # input_dim = 256
        self.fc = nn.Linear(256, self.config['MODEL']['FC']['NUM_CLASSES']).to(self.device)
    
    def forward(self, x, pe):
        out = self.feature(x)
        out = out + pe 
        out = torch.concat((out, pe), dim=2)       
        out = self.lstm(out)        
        out = self.fc(out)
        
        return out

        