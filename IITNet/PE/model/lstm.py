import torch
import torch.nn as nn

class PlainLSTM(nn.Module):

    def __init__(self, config, input_dim, device):
        super(PlainLSTM, self).__init__()
        
        self.device = device
        self.config = config
        self.input_dim = input_dim
        
        self.lstm = nn.LSTM(input_dim, self.config['HIDDEN_DIM'], batch_first = True, 
                            num_layers = self.config['NUM_LAYERS'], bidirectional = self.config['BIDIRECTIONAL'])
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.config['NUM_LAYERS'] * (2 if self.config['BIDIRECTIONAL'] else 1), x.size(0), self.config['HIDDEN_DIM']).to(self.device)
        c0 = torch.zeros(self.config['NUM_LAYERS'] * (2 if self.config['BIDIRECTIONAL'] else 1), x.size(0), self.config['HIDDEN_DIM']).to(self.device)
        return h0, c0 
                    
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
   
        out, hidden = self.lstm(x, (h0, c0))
        out_forward = out[:, -1, :self.config['HIDDEN_DIM']]
        out_backward = out[:, 0, self.config['HIDDEN_DIM']:] 
        
        out = torch.concat((out_forward, out_backward), dim = 1)
        
        return out