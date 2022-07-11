# Copyright 2022 Arkadiusz Choruży


import torch.nn as nn


class ArkNet(nn.Module):
    def __init__(self, num_classes):
        super(ArkNet, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = 34
        self.activation = nn.ReLU()
        self.linear_shape = None

        self.convo_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1),
            self.activation,
            nn.MaxPool2d(3, 2),
            nn.Conv2d(6, 18, 3, 1),
            self.activation,
            nn.Flatten()
            )

        self.linear_layers = nn.Sequential(
            nn.Linear(3042, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.num_classes)
            )
        
    def __call__(self, input):
        return self.forward(input)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convo_layers(x)
        out = self.linear_layers(x)
        return out