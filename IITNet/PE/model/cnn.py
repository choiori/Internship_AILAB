import torch 
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.pool = nn.MaxPool1d(3,2,1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.reshape(-1,1, 6000)
        # print('cnn input: ',x.shape)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        # print('cnn output: ',out.shape)
        out = out.permute(0, 2, 1)
        return out
        