# Copyright 2022 Arkadiusz Choruży


import torch.nn as nn
from torch import flatten
from torchsummary import summary


class SimpleArkNet(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(SimpleArkNet, self).__init__()
        self.num_classes = num_classes

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )

        self.linear_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
            )
        
    def __call__(self, input):
        return self.forward(input)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer0(x)
        x = self.layer1(x)
        out = self.linear_out(x)
        return out

if __name__ == '__name__':
    arch = SimpleArkNet(5, 1)
    print(summary(arch))