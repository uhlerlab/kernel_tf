import torchvision.models as models
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, stride=2, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(16, 32, 3, padding=1, stride=2, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, padding=1, stride=2, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 128, 3, padding=1, stride=2, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 256, 3, padding=1, stride=2, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, 3, padding=1, stride=1, bias=False))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        o = self.conv(x).reshape(-1, 256)
        return self.fc(o)


class PretrainedNet(nn.Module):
    def __init__(self, net, num_classes):
        super(PretrainedNet, self).__init__()
        self.conv = torch.nn.Sequential(*(list(net.children())[:-1]))    
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        o = self.conv(x).reshape(-1, 256)
        return self.fc(o)
    
