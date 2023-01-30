import torch.nn as nn
from ..transformer import Transformer
from ..resnet import ResNetFeature
 
    
class IITNet(nn.Module):
    def __init__(self, config, seq_len, device):
        super(IITNet, self).__init__()
        
        self.device = device
        self.config = config
        self.seq_len = seq_len
        self.resnet_config = config['MODEL']['RESNET']
        self.feature = ResNetFeature(self.resnet_config, seq_len).to(self.device)
        self.transformer = Transformer(config['MODEL']['TRANSFORMER'], device).to(self.device)
    
    def forward(self, x, pe):
        out = self.feature(x)
        out = self.transformer(out, pe)  
                    
        return out
     