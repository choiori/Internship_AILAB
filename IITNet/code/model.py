#%%
import torch
import torch.nn as nn

class IITNet(nn.Module):
    def __init__(self, resnet_config, lstm_config, device):
        super(IITNet, self).__init__()
        self.resnet_config = resnet_config
        self.lstm_config = lstm_config
        self.device = device
        
        self.resnet = ResNetFeature(resnet_config).to(device)
        self.lstm = biLSTM(lstm_config, self.device)

    def forward(self, x):
        out = self.resnet(x)
        out = self.lstm(out)
    
        return out

def conv3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetFeature(nn.Module):

    def __init__(self, config):

        super(ResNetFeature, self).__init__()

        self.layer_config_dict = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        self.inplanes = 16
        self.config = config
        self.config["num_layers"] = self.config["num_layers"]
        self.layers = self.layer_config_dict[config['num_layers']]
        
        if config['num_layers'] == 18 or config['num_layers'] == 34:
            block = BasicBlock
        elif config['num_layers'] == 50 or config['num_layers'] == 101 or config['nun_layers'] == 152:
            block = Bottleneck
        else:
            raise NotImplementedError("num layers '{}' is not in layer config".format(config['num_layers']))

        self.initial_layer = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1))

        self.layer1 = self._make_layer(block, 16, self.layers[0], stride=1, first=True)
        self.layer2 = self._make_layer(block, 16, self.layers[1], stride=2)
        self.layer3 = self._make_layer(block, 32, self.layers[2], stride=2)
        self.layer4 = self._make_layer(block, 32, self.layers[3], stride=2)
        self.maxpool = nn.MaxPool1d(3, 2, 1)

        self.dropout = nn.Dropout(p=config['dropout_rate'])
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, first=False):

        downsample = None
        if (stride != 1 and first is False) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        f_seq = []
        
        for i in range(self.config['seq_len']):
            # tmp = x[:, i]
            # tmp2 = x[:, i].view(x.size(0), 1, -1)
            # print('i: ',i)
            # print('x[:, i]: ',x[:, i].shape)
            # print('x[:, i].view: ',x[:, i].view(x.size(0), 1, -1).shape)
            # f = self.initial_layer(x)  # 첫번째 conv
            f = self.initial_layer(x[:, i].view(x.size(0), 1, -1))
            f = self.layer1(f)
            f = self.layer2(f)
            f = self.maxpool(f)
            f = self.layer3(f)
            f = self.layer4(f)
            f = self.dropout(f)
            f_seq.append(f.permute(0,2,1))
            # out = out.permute(0,2,1)   
        
        out = torch.cat(f_seq, dim=1)    
        return out

        
# class Conv_Block(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, bias):
#         super(Conv_Block, self).__init__()
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias=bias)
#         self.bn = nn.BatchNorm1d(out_channels)
#         self.activation = activation
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
        
#         out = self.conv(x)
#         out = self.bn(out)
        
#         if self.activation == True:    
#             out = self.relu(out)
        
#         return out
        
# class Bottleneck(nn.Module):
#     expansion = 4   
#     def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
#         super(Bottleneck, self).__init__()
        
#         self.conv_block1 = Conv_Block(in_channels, out_channels, 1, 1, 0, True, False)
#         self.conv_block2 = Conv_Block(out_channels, out_channels, 3, stride, 1, True, False)
#         self.conv_block3 = Conv_Block(out_channels, out_channels * self.expansion, 1, 1, 0, False, False)
        
#         self.relu = nn.ReLU()
#         self.downsample = downsample
        
        
#     def forward(self, x):
        
#         identity = x
        
#         out = self.conv_block1(x)
#         out = self.conv_block2(out)
#         out = self.conv_block3(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         out = out + identity
#         out = self.relu(out)
        
#         return out
        
# class ResNet(nn.Module):

#     def __init__(self, config):

#         super(ResNet, self).__init__()
        
#         self.config = config
#         self.layers = config['layers']
#         self.seq_len = config['seq_len']
#         self.block = Bottleneck

        
#         self.conv_block_out_channel = 16 
       
#         self.first_conv = Conv_Block(1, self.conv_block_out_channel, kernel_size=7, stride=2, padding=3, activation=True, bias=False)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) 
#         self.dropout = nn.Dropout(p=0.5)
        
#         self.layer1 = self._make_layer(self.block, 16, 16, self.layers[0])
#         self.layer2 = self._make_layer(self.block, 64, 16, self.layers[1], stride = 2)
#         self.layer3 = self._make_layer(self.block, 64, 32, self.layers[2], stride = 2)
#         self.layer4 = self._make_layer(self.block, 128, 32, self.layers[3], stride = 2)
        
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
                
                
#         #bottleneck 같은 차원끼리 묶음
#     def _make_layer(self, block_type, in_channels, out_channels, block_num, stride = 1):
        
        
#         if (out_channels * block_type.expansion != in_channels) or stride != 1:
#             downsample = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels * block_type.expansion, kernel_size= 1 ,stride = stride, bias=False),
#                 nn.BatchNorm1d(out_channels * block_type.expansion))
#         else:
#             downsample = None
                 
#         layers = [] 
        
#         layers.append(block_type(in_channels, out_channels, stride = stride, downsample = downsample))  # bottleneck 처음부분 (downsampling 필요부분)
        
#         for _ in range(1, block_num):   # downsampling x
#             layers.append(block_type(out_channels * block_type.expansion, out_channels))

#         return nn.Sequential(*layers)
    
 
#     def _forward_impl(self,x):
        
#         out = self.first_conv(x)  # 첫번째 conv
#         out = self.maxpool(out)
        
#         out = self.layer1(out)  # bottleneck
#         out = self.layer2(out)  # bottleneck
#         out = self.maxpool(out)
        
#         out = self.layer3(out)  # bottleneck
#         out = self.layer4(out)  # bottleneck
#         out = self.dropout(out)
        
        
#         out = out.permute(0,2,1)   
        
#         return out
           
#     def forward(self, x):
#         return self._forward_impl(x)    


class biLSTM(nn.Module):
    def __init__(self, config, device):
        super(biLSTM, self).__init__()
        self.config = config
        self.device = device
        self.num_classes = config['num_classes']
        self.num_layers = config['num_layers']
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']

        
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)
    
        
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out_forward = out[:, -1, :self.hidden_size]
        out_back = out[:, 0, self.hidden_size:]
        out = torch.cat((out_forward, out_back), dim=1)
        out = self.fc(out)

        return out


#%%
# from torchinfo import summary

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# resnet_config = {'layers' :  [3, 4, 6, 3] ,  'seq_num': 1}
# lstm_config= {'num_classes' : 5 , 'input_size': 128 , 'hidden_size' : 128, 'num_layers' : 2}

# model = IITNet(resnet_config, lstm_config, device).to(device)

# summary(model, input_size=(1,1, 3000), device='cuda')