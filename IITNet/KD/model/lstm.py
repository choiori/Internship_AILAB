import torch
import torch.nn as nn

class PlainLSTM(nn.Module):

    def __init__(self, config, input_dim, device):
        super(PlainLSTM, self).__init__()
        
        self.device = device
        self.config = config
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(input_dim, self.config['hidden_dim'], batch_first = True, 
                            num_layers = self.config['num_layers'], bidirectional = self.config['bidirectional'])
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.config['num_layers'] * (2 if self.config['bidirectional'] else 1), x.size(0), self.config['hidden_dim']).to(self.device)
        c0 = torch.zeros(self.config['num_layers'] * (2 if self.config['bidirectional'] else 1), x.size(0), self.config['hidden_dim']).to(self.device)
        return h0, c0 
                    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
   
        out, hidden = self.lstm(x, (h0, c0))
        out_forward = out[:, -1, :self.config['hidden_dim']]
        out_backward = out[:, 0, self.config['hidden_dim']:] 
        
        out = torch.concat((out_forward, out_backward), dim = 1)
        
        return out