import torch
import torch.nn as nn
import math
        
        
class Transformer(nn.Module):

    def __init__(self, config, device, fc_out=True):

        super(Transformer, self).__init__()
        # print(config)
        self.num_classes = config['num_classes']
        self.hidden_dim = config['hidden_dim']
        self.nheads = config['nheads']
        self.num_encoder_layers = config['num_encoder_layers']
        self.pool = config['pool']
        self.device =device
        self.pos_encoding = PositionalEncoding(config, self.hidden_dim, device).to(self.device)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nheads, dim_feedforward=config['feedforward_dim'], dropout=0.1 if config['dropout'] else 0.0)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_encoder_layers)

        self.pool = self.pool

        self.fc_out = fc_out
        self.config = config
        if self.config['dropout']:
            self.dropout = nn.Dropout(p=0.5)

        if self.fc_out:
            self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        
        if self.pool == 'attn':
            self.w_ha = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
            self.w_at = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, x): # (B, L, C)
        
        # print(f'x: {x.shape}')                        
        x = x.transpose(0, 1) # (L, B, C)
        pe = pe.transpose(0, 1)
        # x = x + pe
        x = self.pos_encoding(x)   
        # print(f'transformer 들어가기전 : {x.shape}')
        x = self.transformer(x)  # (L, B, C)
        x = x.transpose(0, 1)    # (B, C, L)

        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool == 'last':
            x = x[:, -1]
        elif self.pool == 'attn':
            a_states = torch.tanh(self.w_ha(x))
            alpha = torch.softmax(self.w_at(a_states), dim=1).view(x.size(0), 1, x.size(1))
            x = torch.bmm(alpha, a_states).view(x.size(0), -1)
        elif self.pool == None:
            x = x
        else:
            raise NotImplementedError

        if self.config['dropout']:
            x = self.dropout(x)
        
        if self.fc_out:
            out = self.fc(x)
            return out ### ,x ###
        else:
            return x
        

class PositionalEncoding(nn.Module):
                            ### 128 ###
    def __init__(self, config, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.config = config
        self.input_dim = 128                                                           ### 128 ###
        self.fc = nn.Linear(in_features=self.input_dim, out_features=d_model)
        
        self.max_len = 10000
        print('[INFO] max_len: {}'.format(self.max_len))
            
        self.act_fn = nn.ReLU()
                           ### 1000   , 128
        pe = torch.zeros(self.max_len, d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):   # (L, B, C)
        # print(f'fc 전: {x.shape}')
        x = self.act_fn(self.fc(x))  # ()   
        # print(f'fc 후: {x.shape}')
        pe = self.pe[:x.size(0), :]  # (L, C)
        x = x + pe # ()

        return x

# #############################################################################################
# class PositionalEncoding(nn.Module):
    
#     def __init__(self, config, d_model, device,dropout=0.1 ):
#         super(PositionalEncoding, self).__init__()
#         self.config = config
#         self.dropout = nn.Dropout(p=dropout)
#         # self.input_dim = config['comp_chn']

#         # if self.config['pe_fc']:
#         self.fc = nn.Linear(in_features=d_model, out_features=d_model)

#         # if self.config['pe'] == 'new':
#         #     if self.config['backbone'] == 'VGGV16':
#         #         self.max_len = temporal_cfg[self.config['backbone']][self.config['seq_len'] - 1][self.config['n_anchor'] - 1]
#         #     else:
#         #         self.max_len = temporal_cfg[self.config['backbone']][self.config['n_anchor'] - 1]
        
#         # elif self.config['pe'] == 'old':
#         #     self.max_len = 10000
#         # self.max_len = 5000
#         self.max_len = 47 * 10
#         print('[INFO] max_len: {}'.format(self.max_len))
            
#         # if self.config['pe_activation'] == 'ReLU':
#         self.act_fn = nn.ReLU()
#         # elif self.config['pe_activation'] == 'PReLU':
#         #     self.act_fn = nn.PReLU()

#         pe = torch.zeros(self.max_len, d_model)
#         position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = pe.unsqueeze(0).transpose(0, 1)
#         # self.register_buffer('pe', pe)
#         # print('PE type: ', str(self.config['pe']))
#         self.device = device

#     def forward(self, x):
#  ng(config, self.hidden_dim, device).to(self.device)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
   #     self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=self.nheads, dim_feedforward=config['feedforward_dim'], dropout=0.1 if config['dropout'] else 0.0)
   #     self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=self.num_encoder_layers)
#
    #    self.pool = self.pool
#
#        self.fc_out = fc_out
 #       self.config = config
 #       if self.config['dropout']:       # if self.config['pe_fc']:
#         x = self.act_fn(self.fc(x))

#         # if self.config['pe'] == 'new':
#         #     hop = self.max_len // x.size(0)
         #     pe = self.pe[hop//2::hop, :]

#         #     if pe.shape[0] != x.size(0):
#         #         pe = pe[:x.size(0), :]

#         #     # pe = self.pe[::hop, :]

#         # elif self.config['pe'] == 'old':
#         pe = self.pe[:x.size(0), :]
#         pe = pe.to(self.device)
#         x = x.to(self.device)
#         x = x + pe

#         # if self.config['dropout']:
#         x = self.dropout(x)

#         return x

