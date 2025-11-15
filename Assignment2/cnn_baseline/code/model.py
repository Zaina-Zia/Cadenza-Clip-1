import torch
import torch.nn as nn

class CNN_MFCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 1, n_mfcc, T]
        x = self.conv_layers(x)            # [B, 64, 1, 1]
        x = torch.flatten(x, 1)           # [B, 64]
        x = self.fc(x)                    # [B, 1]
        return torch.sigmoid(x)           # output in [0,1]
